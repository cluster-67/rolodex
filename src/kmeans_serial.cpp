#include "rolodex/kmeans.hpp"

#include "rolodex/distance.hpp"
#include "rolodex/utils.hpp"

#include <H5Cpp.h>
#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <sys/stat.h>
#include <utility>
#include <limits>

SerialKNNAlgorithm::SerialKNNAlgorithm(Dataset *dataset, int num_clusters, bool cache_enabled)
    : KNNAlgorithm(dataset, num_clusters, cache_enabled) {
    dimension_ = dataset_->dim();
    flat_centroids_.resize(static_cast<std::size_t>(num_clusters_) * dimension_);
    centroid_sums_.resize(static_cast<std::size_t>(num_clusters_) * dimension_);
    centroid_counts_.resize(static_cast<std::size_t>(num_clusters_));
    membership_.resize(dataset_->n_points(), -1);
}

void SerialKNNAlgorithm::create_clusters(int update_frequency) {
    const float *points_flat = dataset_->get_flat();
    const std::size_t num_points = dataset_->n_points();

    if (num_points == 0 || num_clusters_ <= 0) {
        return;
    }

    if (cache_enabled_ && load_clusters_from_cache()) {
        std::cout << "Loaded cluster cache from " << get_cache_path() << '\n';
        return;
    }

    for (int c_idx = 0; c_idx < num_clusters_; c_idx++) {
        const int point_idx = static_cast<int>(rand() % num_points);
        std::copy(points_flat + point_idx * dimension_,
                  points_flat + point_idx * dimension_ + dimension_,
                  flat_centroids_.data() + c_idx * dimension_);
        membership_[static_cast<std::size_t>(point_idx)] = c_idx;
    }

    int iters = 0;
    while (true) {
        iters++;
        int membership_change_count = 0;
        for (std::size_t point_idx = 0; point_idx < num_points; point_idx++) {
            int nearest_centroid_idx = find_nearest_centroid(points_flat + point_idx * dimension_);
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

        if (iters % update_frequency == 0) {
            update_centroids();
        }
    }

    if (cache_enabled_) {
        save_clusters_to_cache();
    }
}

void SerialKNNAlgorithm::update_centroids() {
    const float *points_flat = dataset_->get_flat();
    const std::size_t num_points = dataset_->n_points();

    if (num_points == 0) {
        return;
    }

    // 1. Reset existing buffers instead of re-allocating
    std::fill(centroid_sums_.begin(), centroid_sums_.end(), 0.0f);
    std::fill(centroid_counts_.begin(), centroid_counts_.end(), 0.0f);

    for (std::size_t point_idx = 0; point_idx < num_points; point_idx++) {
        const int centroid_idx = membership_[point_idx];
        const std::size_t c_base = static_cast<std::size_t>(centroid_idx) * dimension_;
        const std::size_t p_base = point_idx * dimension_;

        centroid_counts_[static_cast<std::size_t>(centroid_idx)] += 1.0f;
        #pragma omp simd
        for (std::size_t i = 0; i < dimension_; i++) {
            centroid_sums_[c_base + i] += points_flat[p_base + i];
        }
    }

    // 3. Optimized Centroid Calculation
    for (int c_idx = 0; c_idx < num_clusters_; c_idx++) {
        const float count = centroid_counts_[static_cast<std::size_t>(c_idx)];
        if (count > 0.0f) {
            // THE OPTIMIZATION: Calculate reciprocal ONCE
            const float inv_count = 1.0f / count;
            const std::size_t base = static_cast<std::size_t>(c_idx) * dimension_;

            // Use SIMD to multiply by the reciprocal
            #pragma omp simd
            for (std::size_t i = 0; i < dimension_; i++) {
                flat_centroids_[base + i] = centroid_sums_[base + i] * inv_count;
            }
        }
    }
}

int SerialKNNAlgorithm::find_nearest_centroid(const float *query_point) const {
    int nearest_centroid_idx = 0;
    float min_sq_dist = std::numeric_limits<float>::max();

    for (int c = 0; c < num_clusters_; ++c) {
        const float *target_centroid = &flat_centroids_[static_cast<std::size_t>(c) * dimension_];
        float current_dist = 0;

        #pragma omp simd reduction(+:current_dist)
        for (size_t i = 0; i < dimension_; ++i) {
            float diff = query_point[i] - target_centroid[i];
            current_dist += diff * diff;
        }

        if (current_dist < min_sq_dist) {
            min_sq_dist = current_dist;
            nearest_centroid_idx = c;
        }
    }
    return nearest_centroid_idx;
}

QueryResult SerialKNNAlgorithm::query_clusters(const TVector &query, int top_k, int nprobe) const {
    const float *pts_flat = dataset_->get_flat();
    const std::size_t num_points = dataset_->n_points();

    if (num_points == 0 || top_k <= 0 || num_clusters_ <= 0) {
        return QueryResult{};
    }

    const int nprobe_clamped = std::max(1, std::min(nprobe, num_clusters_));

    std::vector<std::pair<float, int>> centroid_dists;
    centroid_dists.reserve(static_cast<std::size_t>(num_clusters_));
    for (int c = 0; c < num_clusters_; ++c) {
        centroid_dists.emplace_back(
            squared_l2(query.data(), &flat_centroids_[static_cast<std::size_t>(c) * dimension_], dimension_), c);
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
        scored.emplace_back(squared_l2(query.data(), pts_flat + idx * dimension_, dimension_),
                            idx);
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
        const std::size_t gidx = scored[i].second;
        TVector neighbor(dimension_);
        std::copy(pts_flat + gidx * dimension_, pts_flat + gidx * dimension_ + dimension_,
                  neighbor.begin());
        result.neighbors.push_back(std::move(neighbor));
        result.distances.push_back(scored[i].first);
    }
    return result;
}

std::vector<int> SerialKNNAlgorithm::find_nearest_points(int centroid_idx, int top_k) const {
    (void)top_k;
    std::vector<int> nearest_points_indices;
    const std::size_t num_points = dataset_->n_points();
    for (std::size_t i = 0; i < num_points; i++) {
        if (membership_[i] == centroid_idx) {
            nearest_points_indices.push_back(static_cast<int>(i));
        }
    }
    return nearest_points_indices;
}

namespace {

const int32_t kCacheFormatVersion = 1;
const char *kCacheAlgorithm = "serial";

bool write_i32_attr(H5::H5Object &obj, const char *name, int32_t value) {
    H5::DataSpace scalar(H5S_SCALAR);
    H5::Attribute attr = obj.createAttribute(name, H5::PredType::STD_I32LE, scalar);
    attr.write(H5::PredType::STD_I32LE, &value);
    return true;
}

bool write_u64_attr(H5::H5Object &obj, const char *name, uint64_t value) {
    H5::DataSpace scalar(H5S_SCALAR);
    H5::Attribute attr = obj.createAttribute(name, H5::PredType::STD_U64LE, scalar);
    attr.write(H5::PredType::STD_U64LE, &value);
    return true;
}

bool write_str_attr(H5::H5Object &obj, const char *name, const std::string &value) {
    H5::StrType str_type(H5::PredType::C_S1, H5T_VARIABLE);
    H5::DataSpace scalar(H5S_SCALAR);
    H5::Attribute attr = obj.createAttribute(name, str_type, scalar);
    attr.write(str_type, value);
    return true;
}

bool read_i32_attr(const H5::H5Object &obj, const char *name, int32_t &out) {
    H5::Attribute attr = obj.openAttribute(name);
    attr.read(H5::PredType::NATIVE_INT32, &out);
    return true;
}

bool read_u64_attr(const H5::H5Object &obj, const char *name, uint64_t &out) {
    H5::Attribute attr = obj.openAttribute(name);
    attr.read(H5::PredType::NATIVE_UINT64, &out);
    return true;
}

bool read_str_attr(const H5::H5Object &obj, const char *name, std::string &out) {
    H5::Attribute attr = obj.openAttribute(name);
    H5::StrType str_type = attr.getStrType();
    attr.read(str_type, out);
    return true;
}

} // namespace

std::string SerialKNNAlgorithm::get_cache_path() const {
    return build_cache_path("serial");
}

bool SerialKNNAlgorithm::load_clusters_from_cache() {
    const std::size_t num_points = dataset_->n_points();
    const std::size_t dim = dataset_->dim();
    if (num_points == 0 || dim == 0 || num_clusters_ <= 0) {
        return false;
    }

    const std::string expected_basename = utils::cache::dataset_basename(dataset_);
    const std::string cache_path = get_cache_path();

    struct stat st;
    if (stat(cache_path.c_str(), &st) != 0) {
        std::cout << "Could not load cluster cache (missing): " << cache_path << '\n';
        return false;
    }

    try {
        H5::H5File in(cache_path.c_str(), H5F_ACC_RDONLY);

        int32_t format_version = 0;
        int32_t cached_num_clusters = 0;
        uint64_t cached_num_points = 0;
        uint64_t cached_dim = 0;
        std::string cached_algorithm;
        std::string cached_dataset_basename;

        read_i32_attr(in, "format_version", format_version);
        read_str_attr(in, "algorithm", cached_algorithm);
        read_i32_attr(in, "num_clusters", cached_num_clusters);
        read_u64_attr(in, "num_points", cached_num_points);
        read_u64_attr(in, "dim", cached_dim);
        read_str_attr(in, "dataset_basename", cached_dataset_basename);

        if (format_version != kCacheFormatVersion || cached_algorithm != kCacheAlgorithm ||
            cached_num_clusters != static_cast<int32_t>(num_clusters_) ||
            cached_num_points != static_cast<uint64_t>(num_points) ||
            cached_dim != static_cast<uint64_t>(dim) ||
            cached_dataset_basename != expected_basename) {
            return false;
        }

        H5::DataSet centroids_ds = in.openDataSet("centroids");
        H5::DataSet membership_ds = in.openDataSet("membership");

        flat_centroids_.assign(static_cast<std::size_t>(num_clusters_) * dim, 0.0f);
        membership_.assign(num_points, -1);

        centroids_ds.read(flat_centroids_.data(), H5::PredType::NATIVE_FLOAT);

        std::vector<int32_t> serialized_membership(num_points);
        membership_ds.read(serialized_membership.data(), H5::PredType::NATIVE_INT32);
        for (std::size_t i = 0; i < num_points; ++i) {
            membership_[i] = static_cast<int>(serialized_membership[i]);
        }
    } catch (const H5::Exception &) {
        return false;
    }

    for (int m : membership_) {
        if (m < 0 || m >= num_clusters_) {
            return false;
        }
    }
    return true;
}

void SerialKNNAlgorithm::save_clusters_to_cache() const {
    const std::size_t num_points = dataset_->n_points();
    const std::size_t dim = dataset_->dim();
    if (num_points == 0 || dim == 0 || num_clusters_ <= 0) {
        return;
    }
    const std::string dataset_basename = utils::cache::dataset_basename(dataset_);

    if (!ensure_cache_root_dir()) {
        std::cerr << "Warning: failed to create cluster cache dir '" << cache_root_dir()
                  << "': errno=" << errno << '\n';
        return;
    }

    try {
        H5::H5File out(get_cache_path().c_str(), H5F_ACC_TRUNC);
        write_i32_attr(out, "format_version", kCacheFormatVersion);
        write_str_attr(out, "algorithm", kCacheAlgorithm);
        write_i32_attr(out, "num_clusters", static_cast<int32_t>(num_clusters_));
        write_u64_attr(out, "num_points", static_cast<uint64_t>(num_points));
        write_u64_attr(out, "dim", static_cast<uint64_t>(dim));
        write_str_attr(out, "dataset_basename", dataset_basename);

        hsize_t centroid_dims[2] = {static_cast<hsize_t>(num_clusters_), static_cast<hsize_t>(dim)};
        H5::DataSpace centroid_space(2, centroid_dims);
        H5::DataSet centroid_ds =
            out.createDataSet("centroids", H5::PredType::IEEE_F32LE, centroid_space);
        centroid_ds.write(flat_centroids_.data(), H5::PredType::NATIVE_FLOAT);

        std::vector<int32_t> serialized_membership(num_points);
        for (std::size_t i = 0; i < num_points; ++i) {
            serialized_membership[i] = static_cast<int32_t>(membership_[i]);
        }
        hsize_t membership_dims[1] = {static_cast<hsize_t>(num_points)};
        H5::DataSpace membership_space(1, membership_dims);
        H5::DataSet membership_ds =
            out.createDataSet("membership", H5::PredType::STD_I32LE, membership_space);
        membership_ds.write(serialized_membership.data(), H5::PredType::NATIVE_INT32);
    } catch (const H5::Exception &) {
        std::cerr << "Warning: failed while writing cluster cache file: " << get_cache_path()
                  << '\n';
        return;
    }
    std::cout << "Saved cluster cache to " << get_cache_path() << '\n';
}

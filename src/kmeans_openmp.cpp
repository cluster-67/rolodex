#include "rolodex/kmeans.hpp"

#include "rolodex/distance.hpp"
#include "rolodex/utils.hpp"

#include <H5Cpp.h>
#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <omp.h>
#include <sstream>

namespace {

const int kDebugSnapshotCadence = 1;
const int32_t kDebugSnapshotFormatVersion = 1;
const char *kDebugAlgorithm = "openmp";

void write_i32_attr(H5::H5Object &obj, const char *name, int32_t value) {
    H5::DataSpace scalar(H5S_SCALAR);
    H5::Attribute attr = obj.createAttribute(name, H5::PredType::STD_I32LE, scalar);
    attr.write(H5::PredType::STD_I32LE, &value);
}

void write_u64_attr(H5::H5Object &obj, const char *name, uint64_t value) {
    H5::DataSpace scalar(H5S_SCALAR);
    H5::Attribute attr = obj.createAttribute(name, H5::PredType::STD_U64LE, scalar);
    attr.write(H5::PredType::STD_U64LE, &value);
}

void write_str_attr(H5::H5Object &obj, const char *name, const std::string &value) {
    H5::StrType str_type(H5::PredType::C_S1, H5T_VARIABLE);
    H5::DataSpace scalar(H5S_SCALAR);
    H5::Attribute attr = obj.createAttribute(name, str_type, scalar);
    attr.write(str_type, value);
}

} // namespace

OpenMPKNNAlgorithm::OpenMPKNNAlgorithm(Dataset *dataset, int num_clusters, bool cache_enabled,
                                       bool debug_enabled)
    : KNNAlgorithm(dataset, num_clusters, cache_enabled), debug_enabled_(debug_enabled) {
    centroids_.resize(static_cast<std::size_t>(num_clusters_));
    membership_.resize(dataset_->get_points().size(), -1);
}

void OpenMPKNNAlgorithm::create_clusters(int update_frequency) {
    std::vector<TVector> &points = dataset_->get_points();
    if (points.empty() || num_clusters_ <= 0) {
        return;
    }

    // Step 1: Main thread randomly picks K initial centroids.
    for (int c_idx = 0; c_idx < num_clusters_; c_idx++) {
        const int point_idx = static_cast<int>(rand() % points.size());
        centroids_[static_cast<std::size_t>(c_idx)] = points[static_cast<std::size_t>(point_idx)];
        membership_[static_cast<std::size_t>(point_idx)] = c_idx;
    }

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

        if (debug_enabled_ && iters % kDebugSnapshotCadence == 0) {
            save_debug_snapshot(iters, false);
        }

        if (membership_change_count == 0) {
            break;
        }

        if (iters % update_frequency == 0) {
            update_centroids();
        }
    }

    if (debug_enabled_) {
        save_debug_snapshot(iters, true);
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

const char *OpenMPKNNAlgorithm::debug_root_dir() {
    return "data/debug";
}

bool OpenMPKNNAlgorithm::ensure_debug_root_dir() const {
    return utils::cache::ensure_dir_recursive(debug_root_dir(), 0775);
}

std::string OpenMPKNNAlgorithm::build_debug_snapshot_path(int iteration, bool is_final) const {
    std::ostringstream out;
    out << debug_root_dir() << "/openmp_dataset_" << utils::cache::dataset_basename(dataset_)
        << "_iter_" << iteration;
    if (is_final) {
        out << "_final";
    }
    out << ".h5";
    return out.str();
}

void OpenMPKNNAlgorithm::save_debug_snapshot(int iteration, bool is_final) const {
    if (!ensure_debug_root_dir()) {
        std::cerr << "[debug] Warning: failed to create debug dir '" << debug_root_dir()
                  << "': errno=" << errno << '\n';
        return;
    }

    const std::vector<TVector> &points = dataset_->get_points();
    if (points.empty()) {
        return;
    }
    const std::size_t num_points = points.size();
    const std::size_t dim = points[0].size();
    const std::string path = build_debug_snapshot_path(iteration, is_final);

    try {
        H5::H5File out(path.c_str(), H5F_ACC_TRUNC);
        write_i32_attr(out, "format_version", kDebugSnapshotFormatVersion);
        write_str_attr(out, "algorithm", kDebugAlgorithm);
        write_str_attr(out, "dataset_basename", utils::cache::dataset_basename(dataset_));
        write_i32_attr(out, "iteration", static_cast<int32_t>(iteration));
        write_i32_attr(out, "is_final", is_final ? 1 : 0);
        write_i32_attr(out, "num_clusters", static_cast<int32_t>(num_clusters_));
        write_u64_attr(out, "num_points", static_cast<uint64_t>(num_points));
        write_u64_attr(out, "dim", static_cast<uint64_t>(dim));

        std::vector<float> centroid_raw(static_cast<std::size_t>(num_clusters_) * dim, 0.0f);
        for (int c = 0; c < num_clusters_; ++c) {
            const std::size_t base = static_cast<std::size_t>(c) * dim;
            for (std::size_t j = 0; j < dim; ++j) {
                centroid_raw[base + j] = centroids_[static_cast<std::size_t>(c)][j];
            }
        }
        hsize_t centroid_dims[2] = {static_cast<hsize_t>(num_clusters_), static_cast<hsize_t>(dim)};
        H5::DataSpace centroid_space(2, centroid_dims);
        H5::DataSet centroid_ds =
            out.createDataSet("centroids", H5::PredType::IEEE_F32LE, centroid_space);
        centroid_ds.write(centroid_raw.data(), H5::PredType::NATIVE_FLOAT);

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
        std::cerr << "[debug] Warning: failed while writing debug snapshot: " << path << '\n';
        return;
    }

    std::cout << "[debug] Saved OpenMP snapshot: iteration=" << iteration
              << " final=" << (is_final ? 1 : 0) << " path=" << path << '\n';
}

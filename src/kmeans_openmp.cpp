#include "rolodex/kmeans.hpp"

#include "rolodex/distance.hpp"
#include "rolodex/timing.hpp"
#include "rolodex/utils.hpp"

#include <H5Cpp.h>
#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <omp.h>
#include <sstream>
#include <utility>

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
    dimension_ = dataset_->dim();
    flat_centroids_.resize(static_cast<std::size_t>(num_clusters_) * dimension_);
    membership_.resize(dataset_->n_points(), -1);
}

void OpenMPKNNAlgorithm::create_clusters(int update_frequency) {
    const float *points_flat = dataset_->get_flat();
    const std::size_t num_points = dataset_->n_points();
    if (num_points == 0 || num_clusters_ <= 0) {
        return;
    }

    cluster_membership_ms_ = 0.0;
    cluster_centroid_update_ms_ = 0.0;
    cluster_membership_iters_ = 0;

    // Step 1: Main thread randomly picks K initial centroids.
    for (int c_idx = 0; c_idx < num_clusters_; c_idx++) {
        const int point_idx = static_cast<int>(rand() % num_points);
        std::copy(points_flat + point_idx * dimension_,
                  points_flat + point_idx * dimension_ + dimension_,
                  flat_centroids_.data() + static_cast<std::size_t>(c_idx) * dimension_);
        membership_[static_cast<std::size_t>(point_idx)] = c_idx;
    }

    int iters = 0;

    while (true) {
        iters++;
        int membership_change_count = 0;

        const auto membership_start = rolodex::timing::SteadyClock::now();
#pragma omp parallel for schedule(static) reduction(+ : membership_change_count)
        for (std::size_t point_idx = 0; point_idx < num_points; point_idx++) {
            const int nearest = find_nearest_centroid(points_flat + point_idx * dimension_);
            if (membership_[point_idx] != nearest) {
                membership_[point_idx] = nearest;
                membership_change_count++;
            }
        }
        const auto membership_end = rolodex::timing::SteadyClock::now();
        cluster_membership_ms_ +=
            rolodex::timing::millis_between(membership_start, membership_end);
        cluster_membership_iters_++;

        std::cout << "Iteration " << iters << " with " << membership_change_count
                  << " membership changes" << '\n';

        if (debug_enabled_ && iters % kDebugSnapshotCadence == 0) {
            save_debug_snapshot(iters, false);
        }

        if (membership_change_count == 0) {
            break;
        }

        if (iters % update_frequency == 0) {
            const auto update_start = rolodex::timing::SteadyClock::now();
            update_centroids();
            const auto update_end = rolodex::timing::SteadyClock::now();
            cluster_centroid_update_ms_ +=
                rolodex::timing::millis_between(update_start, update_end);
        }
    }

    if (debug_enabled_) {
        save_debug_snapshot(iters, true);
    }
}

void OpenMPKNNAlgorithm::update_centroids() {
    const float *points_flat = dataset_->get_flat();
    const std::size_t num_points = dataset_->n_points();
    if (num_points == 0) {
        return;
    }

    const int max_threads = omp_get_max_threads();
    const std::size_t cluster_stride = static_cast<std::size_t>(num_clusters_) * dimension_;

    std::vector<float> local_sums(static_cast<std::size_t>(max_threads) * cluster_stride, 0.0f);
    std::vector<float> local_counts(
        static_cast<std::size_t>(max_threads) * static_cast<std::size_t>(num_clusters_), 0.0f);

#pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        const std::size_t thread_sum_base = static_cast<std::size_t>(tid) * cluster_stride;
        const std::size_t thread_count_base =
            static_cast<std::size_t>(tid) * static_cast<std::size_t>(num_clusters_);
#pragma omp for schedule(static)
        for (std::size_t point_idx = 0; point_idx < num_points; point_idx++) {
            const std::size_t c = static_cast<std::size_t>(membership_[point_idx]);
            const std::size_t c_base = thread_sum_base + c * dimension_;
            const std::size_t p_base = point_idx * dimension_;
            for (std::size_t i = 0; i < dimension_; i++) {
                local_sums[c_base + i] += points_flat[p_base + i];
            }
            local_counts[thread_count_base + c] += 1.0f;
        }
    }

#pragma omp parallel for schedule(static)
    for (int c_idx = 0; c_idx < num_clusters_; c_idx++) {
        const std::size_t c = static_cast<std::size_t>(c_idx);
        float global_count = 0.0f;
        for (int t = 0; t < max_threads; t++) {
            global_count +=
                local_counts[static_cast<std::size_t>(t) * static_cast<std::size_t>(num_clusters_) +
                             c];
        }

        if (global_count > 0.0f) {
            const float inv_count = 1.0f / global_count;
            const std::size_t centroid_base = c * dimension_;
#pragma omp simd
            for (std::size_t i = 0; i < dimension_; i++) {
                float global_sum = 0.0f;
                for (int t = 0; t < max_threads; t++) {
                    global_sum += local_sums[static_cast<std::size_t>(t) * cluster_stride +
                                             centroid_base + i];
                }
                flat_centroids_[centroid_base + i] = global_sum * inv_count;
            }
        }
    }
}

void OpenMPKNNAlgorithm::print_cluster_build_metrics(std::ostream &out) const {
    out << "cluster_build_membership_ms=" << cluster_membership_ms_ << '\n';
    out << "cluster_build_centroid_update_ms=" << cluster_centroid_update_ms_ << '\n';
    out << "cluster_build_membership_iters=" << cluster_membership_iters_ << '\n';
}

int OpenMPKNNAlgorithm::find_nearest_centroid(const float *point) const {
    int nearest_centroid_idx = 0;
    float nearest_sq = std::numeric_limits<float>::max();
    for (int c_idx = 0; c_idx < num_clusters_; c_idx++) {
        const float *centroid =
            flat_centroids_.data() + static_cast<std::size_t>(c_idx) * dimension_;
        float d = 0.0f;
#pragma omp simd reduction(+ : d)
        for (std::size_t i = 0; i < dimension_; i++) {
            const float diff = point[i] - centroid[i];
            d += diff * diff;
        }
        if (d < nearest_sq) {
            nearest_sq = d;
            nearest_centroid_idx = c_idx;
        }
    }
    return nearest_centroid_idx;
}

QueryResult OpenMPKNNAlgorithm::query_clusters(const TVector &query, int top_k, int nprobe) const {
    rolodex::timing::QueryStageGuard guard;
    auto finish = [&](QueryResult result) {
        guard.flush();
        return result;
    };

    const float *pts_flat = dataset_->get_flat();
    const std::size_t num_points = dataset_->n_points();
    if (num_points == 0 || top_k <= 0 || num_clusters_ <= 0) {
        return finish(QueryResult{});
    }

    const int nprobe_clamped = std::max(1, std::min(nprobe, num_clusters_));

    const auto centroid_start = rolodex::timing::SteadyClock::now();
    std::vector<std::pair<float, int>> centroid_dists;
    centroid_dists.reserve(static_cast<std::size_t>(num_clusters_));
    for (int c = 0; c < num_clusters_; ++c) {
        centroid_dists.emplace_back(
            squared_l2(query.data(),
                       flat_centroids_.data() + static_cast<std::size_t>(c) * dimension_,
                       dimension_),
            c);
    }
    std::sort(centroid_dists.begin(), centroid_dists.end(),
              [](const std::pair<float, int> &a, const std::pair<float, int> &b) {
                  if (a.first != b.first) {
                      return a.first < b.first;
                  }
                  return a.second < b.second;
              });
    const auto centroid_end = rolodex::timing::SteadyClock::now();
    guard.stage_.centroid_dist_ms += rolodex::timing::millis_between(centroid_start, centroid_end);

    const auto scan_start = rolodex::timing::SteadyClock::now();
    std::vector<char> probed(static_cast<std::size_t>(num_clusters_), 0);
    for (int p = 0; p < nprobe_clamped; ++p) {
        const int centroid_idx = centroid_dists[static_cast<std::size_t>(p)].second;
        probed[static_cast<std::size_t>(centroid_idx)] = 1;
    }

    const std::size_t k_cap = std::min(static_cast<std::size_t>(top_k), num_points);

    const int max_threads = omp_get_max_threads();
    std::vector<utils::knn::TopKAccumulator> locals;
    locals.reserve(static_cast<std::size_t>(max_threads));
    for (int t = 0; t < max_threads; ++t) {
        locals.emplace_back(k_cap);
    }

#pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        utils::knn::TopKAccumulator &local = locals[static_cast<std::size_t>(tid)];
#pragma omp for schedule(static)
        for (std::size_t idx = 0; idx < num_points; ++idx) {
            const int centroid_idx = membership_[idx];
            if (centroid_idx < 0 || centroid_idx >= num_clusters_) {
                continue;
            }
            if (!probed[static_cast<std::size_t>(centroid_idx)]) {
                continue;
            }
            const float dist = squared_l2(query.data(), pts_flat + idx * dimension_, dimension_);
            if (local.would_accept(dist, idx)) {
                local.push_accepted(dist, idx);
            }
        }
    }

    const auto scan_end = rolodex::timing::SteadyClock::now();
    guard.stage_.scan_ms += rolodex::timing::millis_between(scan_start, scan_end);

    const auto merge_start = rolodex::timing::SteadyClock::now();
    utils::knn::TopKAccumulator merged(k_cap);
    for (const auto &local : locals) {
        for (const auto &entry : local.entries_unsorted()) {
            if (merged.would_accept(entry.first, entry.second)) {
                merged.push_accepted(entry.first, entry.second);
            }
        }
    }
    const auto merge_end = rolodex::timing::SteadyClock::now();
    guard.stage_.openmp_merge_ms += rolodex::timing::millis_between(merge_start, merge_end);

    std::vector<std::pair<float, std::size_t>> scored = merged.extract_sorted();
    if (scored.empty()) {
        return finish(QueryResult{});
    }

    const auto assemble_start = rolodex::timing::SteadyClock::now();
    QueryResult result;
    result.neighbors.reserve(scored.size());
    result.distances.reserve(scored.size());
    for (const auto &entry : scored) {
        const std::size_t gidx = entry.second;
        TVector neighbor(dimension_);
        std::copy(pts_flat + gidx * dimension_, pts_flat + gidx * dimension_ + dimension_,
                  neighbor.begin());
        result.neighbors.push_back(std::move(neighbor));
        result.distances.push_back(entry.first);
    }
    const auto assemble_end = rolodex::timing::SteadyClock::now();
    guard.stage_.result_assemble_ms += rolodex::timing::millis_between(assemble_start, assemble_end);
    return finish(std::move(result));
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

    const std::size_t num_points = dataset_->n_points();
    const std::size_t dim = dataset_->dim();
    if (num_points == 0 || dim == 0) {
        return;
    }
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
        std::cerr << "[debug] Warning: failed while writing debug snapshot: " << path << '\n';
        return;
    }

    std::cout << "[debug] Saved OpenMP snapshot: iteration=" << iteration
              << " final=" << (is_final ? 1 : 0) << " path=" << path << '\n';
}

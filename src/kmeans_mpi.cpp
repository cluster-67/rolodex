#include "rolodex/kmeans.hpp"

#include "rolodex/distance.hpp"

#include <H5Cpp.h>
#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <sys/stat.h>

namespace {

void write_i32_attr(H5::H5Object &obj, const char *name, int32_t value) {
    H5::DataSpace scalar(H5S_SCALAR);
    H5::Attribute attr = obj.createAttribute(name, H5::PredType::STD_I32LE, scalar);
    attr.write(H5::PredType::STD_I32LE, &value);
}

void write_str_attr(H5::H5Object &obj, const char *name, const std::string &value) {
    H5::StrType str_type(H5::PredType::C_S1, H5T_VARIABLE);
    H5::DataSpace scalar(H5S_SCALAR);
    H5::Attribute attr = obj.createAttribute(name, str_type, scalar);
    attr.write(str_type, value);
}

} // namespace

MPIKMeans::MPIKMeans(Dataset *dataset, int num_clusters, bool cache_enabled, int rank, int size)
    : KNNAlgorithm(dataset, num_clusters, cache_enabled), rank_(rank), size_(size), global_n_(0),
      global_offset_(0) {}

void MPIKMeans::create_clusters(int update_frequency) {
    (void)update_frequency;
    // ── Step 1: rank 0 reads the dataset and flattens it ─────────────────────
    int N = 0, D = 0;
    std::vector<float> flat_data;

    if (rank_ == 0) {
        const std::vector<TVector> &pts = dataset_->get_points();
        N = static_cast<int>(pts.size());
        D = static_cast<int>(pts[0].size());
        flat_data.resize(static_cast<std::size_t>(N * D));
        for (int i = 0; i < N; i++) {
            for (int d = 0; d < D; d++) {
                flat_data[static_cast<std::size_t>(i * D + d)] =
                    pts[static_cast<std::size_t>(i)][static_cast<std::size_t>(d)];
            }
        }
    }

    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&D, 1, MPI_INT, 0, MPI_COMM_WORLD);
    global_n_ = N;

    // ── Step 2: rank 0 picks K random centroids from the full dataset ─────────
    // Done before scatter so all N points are available for selection.
    std::vector<float> centroid_flat(static_cast<std::size_t>(num_clusters_ * D), 0.0f);
    if (rank_ == 0) {
        srand(42);
        for (int c = 0; c < num_clusters_; c++) {
            const int idx = rand() % N;
            for (int d = 0; d < D; d++) {
                centroid_flat[static_cast<std::size_t>(c * D + d)] =
                    flat_data[static_cast<std::size_t>(idx * D + d)];
            }
        }
    }
    MPI_Bcast(centroid_flat.data(), num_clusters_ * D, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // ── Compute per-rank slice sizes; handles N not divisible by size_ ────────
    std::vector<int> counts(static_cast<std::size_t>(size_));
    std::vector<int> displs(static_cast<std::size_t>(size_), 0);
    for (int r = 0; r < size_; r++) {
        counts[static_cast<std::size_t>(r)] = (N / size_ + (r < N % size_ ? 1 : 0)) * D;
    }
    for (int r = 1; r < size_; r++) {
        displs[static_cast<std::size_t>(r)] =
            displs[static_cast<std::size_t>(r - 1)] + counts[static_cast<std::size_t>(r - 1)];
    }
    global_offset_ = displs[static_cast<std::size_t>(rank_)] / D;

    const int local_n = counts[static_cast<std::size_t>(rank_)] / D;
    std::vector<float> local_flat(static_cast<std::size_t>(local_n * D));

    MPI_Scatterv(flat_data.data(), counts.data(), displs.data(), MPI_FLOAT, local_flat.data(),
                 local_n * D, MPI_FLOAT, 0, MPI_COMM_WORLD);

    local_points_.assign(static_cast<std::size_t>(local_n), TVector(static_cast<std::size_t>(D)));
    for (int i = 0; i < local_n; i++) {
        for (int d = 0; d < D; d++) {
            local_points_[static_cast<std::size_t>(i)][static_cast<std::size_t>(d)] =
                local_flat[static_cast<std::size_t>(i * D + d)];
        }
    }

    centroids_.assign(static_cast<std::size_t>(num_clusters_),
                      TVector(static_cast<std::size_t>(D)));
    for (int c = 0; c < num_clusters_; c++) {
        for (int d = 0; d < D; d++) {
            centroids_[static_cast<std::size_t>(c)][static_cast<std::size_t>(d)] =
                centroid_flat[static_cast<std::size_t>(c * D + d)];
        }
    }

    local_membership_.assign(static_cast<std::size_t>(local_n), -1);

    const int max_iters = 10000;
    const float convergence_threshold = 0.001f;
    int iters = 0;

    while (true) {
        iters++;

        // ── Step 3: each rank assigns its local points ────────────────────────
        int local_changes = 0;
        for (int i = 0; i < local_n; i++) {
            const int nearest = find_nearest_centroid(local_points_[static_cast<std::size_t>(i)]);
            if (local_membership_[static_cast<std::size_t>(i)] != nearest) {
                local_membership_[static_cast<std::size_t>(i)] = nearest;
                local_changes++;
            }
        }

        // Communication 1: sum local change counts to get the global total.
        int global_changes = 0;
        MPI_Allreduce(&local_changes, &global_changes, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

        if (rank_ == 0) {
            std::cout << "Iteration " << iters << " with " << global_changes
                      << " membership changes\n";
        }

        const float change_ratio =
            static_cast<float>(global_changes) / static_cast<float>(global_n_);
        const bool converged = (iters >= max_iters || change_ratio < convergence_threshold);

        if (iters % 2 == 0 || converged) {
            save_debug_snapshot(iters, converged);
        }

        if (converged) {
            break;
        }

        // ── Steps 4-5: local partial sums → global reduce → new centroids ─────
        update_centroids();
    }
}

void MPIKMeans::update_centroids() {
    const int local_n = static_cast<int>(local_points_.size());
    const int dim = static_cast<int>(local_points_[0].size());

    // Pack layout per cluster: [sum_0 … sum_{D-1} | count].
    // One Allreduce carries both sums and counts, halving collective calls vs.
    // two separate reduces.
    const int buf_size = num_clusters_ * (dim + 1);
    std::vector<float> local_buf(static_cast<std::size_t>(buf_size), 0.0f);
    std::vector<float> global_buf(static_cast<std::size_t>(buf_size), 0.0f);

    // Step 4: each rank accumulates its local partial sums and counts.
    for (int i = 0; i < local_n; i++) {
        const int c = local_membership_[static_cast<std::size_t>(i)];
        const int base = c * (dim + 1);
        for (int d = 0; d < dim; d++) {
            local_buf[static_cast<std::size_t>(base + d)] +=
                local_points_[static_cast<std::size_t>(i)][static_cast<std::size_t>(d)];
        }
        local_buf[static_cast<std::size_t>(base + dim)] += 1.0f;
    }

    // Communication 2: one Allreduce gives every rank the global sums and counts.
    MPI_Allreduce(local_buf.data(), global_buf.data(), buf_size, MPI_FLOAT, MPI_SUM,
                  MPI_COMM_WORLD);

    // Step 5: every rank recomputes the same new centroids independently.
    for (int c = 0; c < num_clusters_; c++) {
        const int base = c * (dim + 1);
        const float count = global_buf[static_cast<std::size_t>(base + dim)];
        if (count > 0.0f) {
            for (int d = 0; d < dim; d++) {
                centroids_[static_cast<std::size_t>(c)][static_cast<std::size_t>(d)] =
                    global_buf[static_cast<std::size_t>(base + d)] / count;
            }
        }
    }
}

int MPIKMeans::find_nearest_centroid(const TVector &point) const {
    int nearest = 0;
    float best = squared_l2(point, centroids_[0]);
    for (int c = 1; c < num_clusters_; c++) {
        const float d = squared_l2(point, centroids_[static_cast<std::size_t>(c)]);
        if (d < best) {
            best = d;
            nearest = c;
        }
    }
    return nearest;
}

QueryResult MPIKMeans::query_clusters(const TVector &query, int top_k, int nprobe) const {
    (void)nprobe;
    if (top_k <= 0 || local_points_.empty() || num_clusters_ <= 0) {
        return QueryResult{};
    }

    const int nearest_centroid_idx = find_nearest_centroid(query);
    const std::vector<int> candidate_indices = find_nearest_points(nearest_centroid_idx, top_k);
    if (candidate_indices.empty()) {
        return QueryResult{};
    }

    std::vector<std::pair<float, std::size_t>> scored;
    scored.reserve(candidate_indices.size());
    for (int idx : candidate_indices) {
        const std::size_t i = static_cast<std::size_t>(idx);
        if (i >= local_points_.size()) {
            continue;
        }
        scored.emplace_back(squared_l2(query, local_points_[i]), i);
    }
    if (scored.empty()) {
        return QueryResult{};
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
        result.neighbors.push_back(local_points_[scored[i].second]);
        result.distances.push_back(scored[i].first);
    }
    return result;
}

std::vector<int> MPIKMeans::find_nearest_points(int centroid_idx, int top_k) const {
    (void)top_k;
    std::vector<int> nearest_points_indices;
    for (std::size_t i = 0; i < local_points_.size(); i++) {
        if (local_membership_[i] == centroid_idx) {
            nearest_points_indices.push_back(static_cast<int>(i));
        }
    }
    return nearest_points_indices;
}

const char *MPIKMeans::debug_root_dir() { return "data/debug/mpi"; }

bool MPIKMeans::ensure_debug_root_dir() const {
    mkdir("data", 0755);
    mkdir("data/debug", 0755);
    const int rc = mkdir(debug_root_dir(), 0755);
    return rc == 0 || errno == EEXIST;
}

std::string MPIKMeans::build_debug_snapshot_path(int iteration, bool is_final) const {
    std::ostringstream oss;
    oss << debug_root_dir() << "/mpi_rank_" << rank_ << "_iter_"
        << std::setw(6) << std::setfill('0') << iteration;
    if (is_final) oss << "_final";
    oss << ".h5";
    return oss.str();
}

void MPIKMeans::save_debug_snapshot(int iteration, bool is_final) const {
    if (!ensure_debug_root_dir()) {
        std::cerr << "[debug] Warning: failed to create debug dir '" << debug_root_dir()
                  << "': errno=" << errno << '\n';
        return;
    }

    const int local_n = static_cast<int>(local_membership_.size());
    const int dim = static_cast<int>(centroids_[0].size());
    const std::string path = build_debug_snapshot_path(iteration, is_final);

    try {
        H5::H5File out(path.c_str(), H5F_ACC_TRUNC);
        write_i32_attr(out, "format_version", 1);
        write_str_attr(out, "algorithm", std::string("mpi"));
        write_i32_attr(out, "iteration", static_cast<int32_t>(iteration));
        write_i32_attr(out, "is_final", is_final ? 1 : 0);
        write_i32_attr(out, "num_clusters", static_cast<int32_t>(num_clusters_));
        write_i32_attr(out, "rank", static_cast<int32_t>(rank_));
        write_i32_attr(out, "num_ranks", static_cast<int32_t>(size_));
        write_i32_attr(out, "global_offset", static_cast<int32_t>(global_offset_));
        write_i32_attr(out, "local_n", static_cast<int32_t>(local_n));
        write_i32_attr(out, "global_n", static_cast<int32_t>(global_n_));

        std::vector<float> centroid_raw(static_cast<std::size_t>(num_clusters_) * dim, 0.0f);
        for (int c = 0; c < num_clusters_; ++c) {
            const std::size_t base = static_cast<std::size_t>(c) * dim;
            for (int d = 0; d < dim; ++d) {
                centroid_raw[base + static_cast<std::size_t>(d)] =
                    centroids_[static_cast<std::size_t>(c)][static_cast<std::size_t>(d)];
            }
        }
        hsize_t centroid_dims[2] = {static_cast<hsize_t>(num_clusters_),
                                    static_cast<hsize_t>(dim)};
        H5::DataSpace centroid_space(2, centroid_dims);
        H5::DataSet centroid_ds =
            out.createDataSet("centroids", H5::PredType::IEEE_F32LE, centroid_space);
        centroid_ds.write(centroid_raw.data(), H5::PredType::NATIVE_FLOAT);

        std::vector<int32_t> serialized_membership(static_cast<std::size_t>(local_n));
        for (int i = 0; i < local_n; ++i) {
            serialized_membership[static_cast<std::size_t>(i)] =
                static_cast<int32_t>(local_membership_[static_cast<std::size_t>(i)]);
        }
        hsize_t membership_dims[1] = {static_cast<hsize_t>(local_n)};
        H5::DataSpace membership_space(1, membership_dims);
        H5::DataSet membership_ds =
            out.createDataSet("membership", H5::PredType::STD_I32LE, membership_space);
        membership_ds.write(serialized_membership.data(), H5::PredType::NATIVE_INT32);
    } catch (const H5::Exception &) {
        std::cerr << "[debug] Warning: failed while writing MPI debug snapshot: " << path << '\n';
        return;
    }

    std::cout << "[debug] Saved MPI snapshot: rank=" << rank_ << " iteration=" << iteration
              << " final=" << (is_final ? 1 : 0) << " path=" << path << '\n';
}
#include "rolodex/kmeans.hpp"

#include "rolodex/dataset.hpp"
#include "rolodex/distance.hpp"
#include "rolodex/utils.hpp"

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <utility>
#include <vector>

MPIKMeans::MPIKMeans(Dataset *dataset, int num_clusters, bool cache_enabled, int rank, int size)
    : KNNAlgorithm(dataset, num_clusters, cache_enabled), rank_(rank), size_(size), global_n_(0),
      global_point_offset_(0), dimension_(0) {}

void MPIKMeans::create_clusters(int update_frequency) {
    // Step 1: rank 0 reads the dataset and flattens it
    int N = 0, D = 0;
    std::vector<float> flat_data;

    if (rank_ == 0) {
        N = static_cast<int>(dataset_->n_points());
        D = static_cast<int>(dataset_->dim());
        flat_data.resize(static_cast<std::size_t>(N * D));
        std::copy(dataset_->get_flat(), dataset_->get_flat() + static_cast<std::size_t>(N) * D,
                  flat_data.begin());
    }

    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&D, 1, MPI_INT, 0, MPI_COMM_WORLD);
    global_n_ = N;
    dimension_ = D;
    if (N <= 0 || D <= 0 || num_clusters_ <= 0) {
        return;
    }

    // Per-rank point counts (same formula as Scatterv counts)
    int prefix_points = 0;
    for (int rr = 0; rr < rank_; ++rr) {
        prefix_points += N / size_ + (rr < N % size_ ? 1 : 0);
    }
    global_point_offset_ = prefix_points;

    // Step 2: rank 0 picks K random centroids from the full dataset (RNG seeded in main).
    std::vector<float> centroid_flat(static_cast<std::size_t>(num_clusters_ * D), 0.0f);
    if (rank_ == 0) {
        for (int c = 0; c < num_clusters_; c++) {
            const int idx = rand() % N;
            for (int d = 0; d < D; d++) {
                centroid_flat[static_cast<std::size_t>(c * D + d)] =
                    flat_data[static_cast<std::size_t>(idx * D + d)];
            }
        }
    }
    MPI_Bcast(centroid_flat.data(), num_clusters_ * D, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // Per-rank slice sizes; handles N not divisible by size_
    std::vector<int> counts(static_cast<std::size_t>(size_));
    std::vector<int> displs(static_cast<std::size_t>(size_), 0);
    for (int r = 0; r < size_; r++) {
        counts[static_cast<std::size_t>(r)] = (N / size_ + (r < N % size_ ? 1 : 0)) * D;
    }
    for (int r = 1; r < size_; r++) {
        displs[static_cast<std::size_t>(r)] =
            displs[static_cast<std::size_t>(r - 1)] + counts[static_cast<std::size_t>(r - 1)];
    }

    const int local_n = counts[static_cast<std::size_t>(rank_)] / D;
    local_points_flat_.assign(static_cast<std::size_t>(local_n * D), 0.0f);

    MPI_Scatterv(flat_data.data(), counts.data(), displs.data(), MPI_FLOAT,
                 local_points_flat_.data(), local_n * D, MPI_FLOAT, 0, MPI_COMM_WORLD);

    flat_centroids_.assign(centroid_flat.begin(), centroid_flat.end());

    local_membership_.assign(static_cast<std::size_t>(local_n), -1);

    int iters = 0;

    while (true) {
        iters++;

        int local_changes = 0;
        for (int i = 0; i < local_n; i++) {
            const int nearest =
                find_nearest_centroid(local_points_flat_.data() + static_cast<std::size_t>(i) * D);
            if (local_membership_[static_cast<std::size_t>(i)] != nearest) {
                local_membership_[static_cast<std::size_t>(i)] = nearest;
                local_changes++;
            }
        }

        int global_changes = 0;
        MPI_Allreduce(&local_changes, &global_changes, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

        if (rank_ == 0) {
            std::cout << "Iteration " << iters << " with " << global_changes
                      << " membership changes\n";
        }

        if (global_changes == 0) {
            break;
        }

        if (iters % update_frequency == 0) {
            update_centroids();
        }
    }
}

void MPIKMeans::update_centroids() {
    const int local_n =
        dimension_ > 0
            ? static_cast<int>(local_points_flat_.size() / static_cast<std::size_t>(dimension_))
            : 0;
    const int dim = dimension_;

    const int buf_size = num_clusters_ * (dim + 1);
    std::vector<float> local_buf(static_cast<std::size_t>(buf_size), 0.0f);
    std::vector<float> global_buf(static_cast<std::size_t>(buf_size), 0.0f);

    for (int i = 0; i < local_n; i++) {
        const int c = local_membership_[static_cast<std::size_t>(i)];
        const int base = c * (dim + 1);
#pragma omp simd
        for (int d = 0; d < dim; d++) {
            local_buf[static_cast<std::size_t>(base + d)] +=
                local_points_flat_[static_cast<std::size_t>(i) * static_cast<std::size_t>(dim) +
                                   static_cast<std::size_t>(d)];
        }
        local_buf[static_cast<std::size_t>(base + dim)] += 1.0f;
    }

    MPI_Allreduce(local_buf.data(), global_buf.data(), buf_size, MPI_FLOAT, MPI_SUM,
                  MPI_COMM_WORLD);

    for (int c = 0; c < num_clusters_; c++) {
        const int base = c * (dim + 1);
        const float count = global_buf[static_cast<std::size_t>(base + dim)];
        if (count > 0.0f) {
            const float inv_count = 1.0f / count;
#pragma omp simd
            for (int d = 0; d < dim; d++) {
                flat_centroids_[static_cast<std::size_t>(c) * static_cast<std::size_t>(dim) +
                                static_cast<std::size_t>(d)] =
                    global_buf[static_cast<std::size_t>(base + d)] * inv_count;
            }
        }
    }
}

int MPIKMeans::find_nearest_centroid(const float *point) const {
    int nearest = 0;
    float best = std::numeric_limits<float>::max();
    for (int c = 0; c < num_clusters_; c++) {
        const float *centroid = flat_centroids_.data() +
                                static_cast<std::size_t>(c) * static_cast<std::size_t>(dimension_);
        float d = 0.0f;
#pragma omp simd reduction(+ : d)
        for (int i = 0; i < dimension_; i++) {
            const float diff =
                point[static_cast<std::size_t>(i)] - centroid[static_cast<std::size_t>(i)];
            d += diff * diff;
        }
        if (d < best) {
            best = d;
            nearest = c;
        }
    }
    return nearest;
}

int MPIKMeans::vector_dim() const {
    return dimension_;
}

QueryResult MPIKMeans::query_clusters(const TVector &query, int top_k, int nprobe) const {
    int tk = 0;
    int np = 0;
    if (rank_ == 0) {
        tk = top_k;
        np = nprobe;
    }
    MPI_Bcast(&tk, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&np, 1, MPI_INT, 0, MPI_COMM_WORLD);

    const int dim = vector_dim();
    std::vector<float> qbuf(static_cast<std::size_t>(std::max(0, dim)), 0.0f);
    if (rank_ == 0) {
        for (int i = 0; i < dim; ++i) {
            qbuf[static_cast<std::size_t>(i)] = query[static_cast<std::size_t>(i)];
        }
    }
    if (dim > 0) {
        MPI_Bcast(qbuf.data(), dim, MPI_FLOAT, 0, MPI_COMM_WORLD);
    }

    if (tk <= 0 || num_clusters_ <= 0 || dim <= 0) {
        return QueryResult{};
    }

    const int nprobe_clamped = std::max(1, std::min(np, num_clusters_));

    std::vector<std::pair<float, int>> centroid_dists;
    centroid_dists.reserve(static_cast<std::size_t>(num_clusters_));
    for (int c = 0; c < num_clusters_; ++c) {
        centroid_dists.emplace_back(
            squared_l2(qbuf.data(),
                       flat_centroids_.data() +
                           static_cast<std::size_t>(c) * static_cast<std::size_t>(dim),
                       static_cast<std::size_t>(dim)),
            c);
    }
    std::sort(centroid_dists.begin(), centroid_dists.end(),
              [](const std::pair<float, int> &a, const std::pair<float, int> &b) {
                  if (a.first != b.first) {
                      return a.first < b.first;
                  }
                  return a.second < b.second;
              });

    std::vector<char> probed(static_cast<std::size_t>(num_clusters_), 0);
    for (int p = 0; p < nprobe_clamped; ++p) {
        const int centroid_idx = centroid_dists[static_cast<std::size_t>(p)].second;
        probed[static_cast<std::size_t>(centroid_idx)] = 1;
    }

    const std::size_t local_n =
        dim > 0 ? local_points_flat_.size() / static_cast<std::size_t>(dim) : 0;
    const std::size_t k_cap = std::min(static_cast<std::size_t>(tk), local_n);
    utils::knn::TopKAccumulator topk(k_cap);
    for (std::size_t liu = 0; liu < local_n; ++liu) {
        const int centroid_idx = local_membership_[liu];
        if (centroid_idx < 0 || centroid_idx >= num_clusters_) {
            continue;
        }
        if (!probed[static_cast<std::size_t>(centroid_idx)]) {
            continue;
        }
        const float d =
            squared_l2(qbuf.data(), local_points_flat_.data() + liu * static_cast<std::size_t>(dim),
                       static_cast<std::size_t>(dim));
        const std::size_t gidx = static_cast<std::size_t>(global_point_offset_) + liu;
        if (topk.would_accept(d, gidx)) {
            topk.push_accepted(d, gidx);
        }
    }

    std::vector<std::pair<float, std::size_t>> scored = topk.extract_sorted();
    const int local_count = static_cast<int>(scored.size());
    std::vector<float> send_d(static_cast<std::size_t>(local_count));
    std::vector<int> send_g(static_cast<std::size_t>(local_count));
    for (int i = 0; i < local_count; ++i) {
        send_d[static_cast<std::size_t>(i)] = scored[static_cast<std::size_t>(i)].first;
        send_g[static_cast<std::size_t>(i)] =
            static_cast<int>(scored[static_cast<std::size_t>(i)].second);
    }
    std::vector<int> recvcounts;
    if (rank_ == 0) {
        recvcounts.resize(static_cast<std::size_t>(size_));
    }
    MPI_Gather(&local_count, 1, MPI_INT, rank_ == 0 ? recvcounts.data() : nullptr, 1, MPI_INT, 0,
               MPI_COMM_WORLD);

    std::vector<int> displs;
    int total_recv = 0;
    if (rank_ == 0) {
        displs.resize(static_cast<std::size_t>(size_));
        for (int r = 0; r < size_; ++r) {
            displs[static_cast<std::size_t>(r)] = total_recv;
            total_recv += recvcounts[static_cast<std::size_t>(r)];
        }
    }

    std::vector<float> alldist;
    std::vector<int> allg;
    if (rank_ == 0 && total_recv > 0) {
        alldist.resize(static_cast<std::size_t>(total_recv));
        allg.resize(static_cast<std::size_t>(total_recv));
    }

    MPI_Gatherv(send_d.data(), local_count, MPI_FLOAT, rank_ == 0 ? alldist.data() : nullptr,
                rank_ == 0 ? recvcounts.data() : nullptr, rank_ == 0 ? displs.data() : nullptr,
                MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Gatherv(send_g.data(), local_count, MPI_INT, rank_ == 0 ? allg.data() : nullptr,
                rank_ == 0 ? recvcounts.data() : nullptr, rank_ == 0 ? displs.data() : nullptr,
                MPI_INT, 0, MPI_COMM_WORLD);

    QueryResult result;
    if (rank_ != 0) {
        return result;
    }

    if (total_recv <= 0) {
        return result;
    }

    utils::knn::TopKAccumulator merged_topk(static_cast<std::size_t>(tk));
    for (int i = 0; i < total_recv; ++i) {
        const float d = alldist[static_cast<std::size_t>(i)];
        const std::size_t gidx = static_cast<std::size_t>(allg[static_cast<std::size_t>(i)]);
        if (merged_topk.would_accept(d, gidx)) {
            merged_topk.push_accepted(d, gidx);
        }
    }
    std::vector<std::pair<float, std::size_t>> merged = merged_topk.extract_sorted();
    const std::size_t k_out = merged.size();

    const float *pts_flat = dataset_->get_flat();
    const std::size_t num_points = dataset_->n_points();
    result.neighbors.reserve(k_out);
    result.distances.reserve(k_out);
    for (std::size_t i = 0; i < k_out; ++i) {
        const std::size_t gix = merged[i].second;
        if (gix >= num_points) {
            continue;
        }
        TVector neighbor(static_cast<std::size_t>(dim));
        std::copy(pts_flat + gix * static_cast<std::size_t>(dim),
                  pts_flat + gix * static_cast<std::size_t>(dim) + static_cast<std::size_t>(dim),
                  neighbor.begin());
        result.neighbors.push_back(std::move(neighbor));
        result.distances.push_back(merged[i].first);
    }
    return result;
}

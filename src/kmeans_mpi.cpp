#include "rolodex/kmeans.hpp"

#include "rolodex/dataset.hpp"
#include "rolodex/distance.hpp"

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <utility>
#include <vector>

MPIKMeans::MPIKMeans(Dataset *dataset, int num_clusters, bool cache_enabled, int rank, int size)
    : KNNAlgorithm(dataset, num_clusters, cache_enabled), rank_(rank), size_(size), global_n_(0),
      global_point_offset_(0) {}

void MPIKMeans::create_clusters(int update_frequency) {
    // Step 1: rank 0 reads the dataset and flattens it
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

    int iters = 0;

    while (true) {
        iters++;

        int local_changes = 0;
        for (int i = 0; i < local_n; i++) {
            const int nearest = find_nearest_centroid(local_points_[static_cast<std::size_t>(i)]);
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
    const int local_n = static_cast<int>(local_points_.size());
    const int dim = local_n > 0 ? static_cast<int>(local_points_[0].size())
                                : static_cast<int>(centroids_[0].size());

    const int buf_size = num_clusters_ * (dim + 1);
    std::vector<float> local_buf(static_cast<std::size_t>(buf_size), 0.0f);
    std::vector<float> global_buf(static_cast<std::size_t>(buf_size), 0.0f);

    for (int i = 0; i < local_n; i++) {
        const int c = local_membership_[static_cast<std::size_t>(i)];
        const int base = c * (dim + 1);
        for (int d = 0; d < dim; d++) {
            local_buf[static_cast<std::size_t>(base + d)] +=
                local_points_[static_cast<std::size_t>(i)][static_cast<std::size_t>(d)];
        }
        local_buf[static_cast<std::size_t>(base + dim)] += 1.0f;
    }

    MPI_Allreduce(local_buf.data(), global_buf.data(), buf_size, MPI_FLOAT, MPI_SUM,
                  MPI_COMM_WORLD);

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

int MPIKMeans::vector_dim() const {
    if (centroids_.empty()) {
        return 0;
    }
    return static_cast<int>(centroids_[0].size());
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

    TVector q(static_cast<std::size_t>(dim));
    for (int i = 0; i < dim; ++i) {
        q[static_cast<std::size_t>(i)] = qbuf[static_cast<std::size_t>(i)];
    }

    if (tk <= 0 || num_clusters_ <= 0 || dim <= 0) {
        return QueryResult{};
    }

    const int nprobe_clamped = std::max(1, std::min(np, num_clusters_));

    std::vector<std::pair<float, int>> centroid_dists;
    centroid_dists.reserve(static_cast<std::size_t>(num_clusters_));
    for (int c = 0; c < num_clusters_; ++c) {
        centroid_dists.emplace_back(squared_l2(q, centroids_[static_cast<std::size_t>(c)]), c);
    }
    std::sort(centroid_dists.begin(), centroid_dists.end(),
              [](const std::pair<float, int> &a, const std::pair<float, int> &b) {
                  if (a.first != b.first) {
                      return a.first < b.first;
                  }
                  return a.second < b.second;
              });

    // All local points in probed clusters; then keep only local top-k for gather (sufficient for
    // exact global top-k merge on rank 0).
    std::vector<std::pair<float, std::size_t>> scored;
    for (int p = 0; p < nprobe_clamped; ++p) {
        const int centroid_idx = centroid_dists[static_cast<std::size_t>(p)].second;
        const std::vector<int> local_indices = find_nearest_points(centroid_idx, tk);
        for (int li : local_indices) {
            const std::size_t liu = static_cast<std::size_t>(li);
            const float d = squared_l2(q, local_points_[liu]);
            const int gidx = global_point_offset_ + li;
            scored.emplace_back(d, static_cast<std::size_t>(gidx));
        }
    }

    const std::size_t k_local =
        std::min(static_cast<std::size_t>(tk), scored.size());
    if (k_local > 0) {
        std::partial_sort(
            scored.begin(), scored.begin() + static_cast<long>(k_local), scored.end(),
            [](const std::pair<float, std::size_t> &a, const std::pair<float, std::size_t> &b) {
                if (a.first != b.first) {
                    return a.first < b.first;
                }
                return a.second < b.second;
            });
    }

    const int local_count = static_cast<int>(k_local);
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

    std::vector<std::pair<float, std::size_t>> merged;
    merged.reserve(static_cast<std::size_t>(total_recv));
    for (int i = 0; i < total_recv; ++i) {
        merged.emplace_back(alldist[static_cast<std::size_t>(i)],
                            static_cast<std::size_t>(allg[static_cast<std::size_t>(i)]));
    }

    const std::size_t k_out = std::min(static_cast<std::size_t>(tk), merged.size());
    std::partial_sort(
        merged.begin(), merged.begin() + static_cast<long>(k_out), merged.end(),
        [](const std::pair<float, std::size_t> &a, const std::pair<float, std::size_t> &b) {
            if (a.first != b.first) {
                return a.first < b.first;
            }
            return a.second < b.second;
        });

    const std::vector<TVector> &pts = dataset_->get_points();
    result.neighbors.reserve(k_out);
    result.distances.reserve(k_out);
    for (std::size_t i = 0; i < k_out; ++i) {
        const std::size_t gix = merged[i].second;
        if (gix >= pts.size()) {
            continue;
        }
        result.neighbors.push_back(pts[gix]);
        result.distances.push_back(merged[i].first);
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

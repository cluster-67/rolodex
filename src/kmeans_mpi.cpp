#include "rolodex/kmeans.hpp"

#include "rolodex/distance.hpp"

#include <cstdlib>
#include <iostream>

MPIKMeans::MPIKMeans(Dataset *dataset, int num_clusters, int rank, int size)
    : KNNAlgorithm(dataset, num_clusters), rank_(rank), size_(size), global_n_(0) {}

void MPIKMeans::create_clusters() {
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

    const int local_n = counts[static_cast<std::size_t>(rank_)] / D;
    std::vector<float> local_flat(static_cast<std::size_t>(local_n * D));

    MPI_Scatterv(flat_data.data(), counts.data(), displs.data(), MPI_FLOAT,
                 local_flat.data(), local_n * D, MPI_FLOAT, 0, MPI_COMM_WORLD);

    local_points_.assign(static_cast<std::size_t>(local_n), TVector(static_cast<std::size_t>(D)));
    for (int i = 0; i < local_n; i++) {
        for (int d = 0; d < D; d++) {
            local_points_[static_cast<std::size_t>(i)][static_cast<std::size_t>(d)] =
                local_flat[static_cast<std::size_t>(i * D + d)];
        }
    }

    centroids_.assign(static_cast<std::size_t>(num_clusters_), TVector(static_cast<std::size_t>(D)));
    for (int c = 0; c < num_clusters_; c++) {
        for (int d = 0; d < D; d++) {
            centroids_[static_cast<std::size_t>(c)][static_cast<std::size_t>(d)] =
                centroid_flat[static_cast<std::size_t>(c * D + d)];
        }
    }

    local_membership_.assign(static_cast<std::size_t>(local_n), -1);

    const int   max_iters             = 10000;
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
            std::cout << "Iteration " << iters
                      << " with " << global_changes << " membership changes\n";
        }

        const float change_ratio =
            static_cast<float>(global_changes) / static_cast<float>(global_n_);

        if (iters >= max_iters || change_ratio < convergence_threshold) {
            break;
        }

        // ── Steps 4-5: local partial sums → global reduce → new centroids ─────
        update_centroids();
    }
}

void MPIKMeans::update_centroids() {
    const int local_n = static_cast<int>(local_points_.size());
    const int dim     = static_cast<int>(local_points_[0].size());

    // Pack layout per cluster: [sum_0 … sum_{D-1} | count].
    // One Allreduce carries both sums and counts, halving collective calls vs.
    // two separate reduces.
    const int buf_size = num_clusters_ * (dim + 1);
    std::vector<float> local_buf(static_cast<std::size_t>(buf_size), 0.0f);
    std::vector<float> global_buf(static_cast<std::size_t>(buf_size), 0.0f);

    // Step 4: each rank accumulates its local partial sums and counts.
    for (int i = 0; i < local_n; i++) {
        const int c    = local_membership_[static_cast<std::size_t>(i)];
        const int base = c * (dim + 1);
        for (int d = 0; d < dim; d++) {
            local_buf[static_cast<std::size_t>(base + d)] +=
                local_points_[static_cast<std::size_t>(i)][static_cast<std::size_t>(d)];
        }
        local_buf[static_cast<std::size_t>(base + dim)] += 1.0f;
    }

    // Communication 2: one Allreduce gives every rank the global sums and counts.
    MPI_Allreduce(local_buf.data(), global_buf.data(), buf_size,
                  MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

    // Step 5: every rank recomputes the same new centroids independently.
    for (int c = 0; c < num_clusters_; c++) {
        const int   base  = c * (dim + 1);
        const float count = global_buf[static_cast<std::size_t>(base + dim)];
        if (count > 0.0f) {
            for (int d = 0; d < dim; d++) {
                centroids_[static_cast<std::size_t>(c)][static_cast<std::size_t>(d)] =
                    global_buf[static_cast<std::size_t>(base + d)] / count;
            }
        }
    }
}

int MPIKMeans::find_nearest_centroid(TVector &point) {
    int   nearest = 0;
    float best    = squared_l2(point, centroids_[0]);
    for (int c = 1; c < num_clusters_; c++) {
        const float d = squared_l2(point, centroids_[static_cast<std::size_t>(c)]);
        if (d < best) {
            best    = d;
            nearest = c;
        }
    }
    return nearest;
}

std::vector<TVector> MPIKMeans::query_clusters(TVector &query, int top_k) {
    (void)top_k;
    const int nearest_centroid_idx = find_nearest_centroid(query);
    std::vector<int> nearest_points_indices = find_nearest_points(nearest_centroid_idx, top_k);
    std::vector<TVector> nearest_points;
    nearest_points.reserve(nearest_points_indices.size());
    for (int idx : nearest_points_indices) {
        nearest_points.push_back(local_points_[static_cast<std::size_t>(idx)]);
    }
    return nearest_points;
}

std::vector<int> MPIKMeans::find_nearest_points(int centroid_idx, int top_k) {
    (void)top_k;
    std::vector<int> nearest_points_indices;
    for (std::size_t i = 0; i < local_points_.size(); i++) {
        if (local_membership_[i] == centroid_idx) {
            nearest_points_indices.push_back(static_cast<int>(i));
        }
    }
    return nearest_points_indices;
}
#pragma once

#include "rolodex/dataset.hpp"
#include "rolodex/types.hpp"

#include <mpi.h>
#include <cstddef>
#include <vector>

class KNNAlgorithm {
  protected:
    Dataset *dataset_;
    int      num_clusters_;

  public:
    KNNAlgorithm(Dataset *dataset, int num_clusters);
    virtual ~KNNAlgorithm() = default;
};

// ── Serial — flat memory layout ───────────────────────────────────────────────
class SerialKNNAlgorithm : public KNNAlgorithm {
  private:
    std::vector<float> centroids_flat_;   // k * dim, row-major
    std::vector<float> centroid_sums_;    // reused across iterations
    std::vector<float> centroid_counts_;  // reused across iterations
    std::vector<int>   membership_;

    // Takes raw pointer — picks the float* overload of squared_l2
    int find_nearest_centroid(const float* point, std::size_t dim);

  public:
    SerialKNNAlgorithm(Dataset *dataset, int num_clusters);

    void create_clusters();
    void update_centroids();
    std::vector<int> find_nearest_points(int centroid_idx, int top_k);
};

// ── OpenMP — unchanged interface ──────────────────────────────────────────────
class OpenMPKNNAlgorithm : public KNNAlgorithm {
  private:
    std::vector<TVector> centroids_;
    std::vector<int>     membership_;

  public:
    OpenMPKNNAlgorithm(Dataset *dataset, int num_clusters);

    void create_clusters(int update_frequency);
    void update_centroids();
    int  find_nearest_centroid(TVector &point);
    std::vector<TVector> query_clusters(TVector &query, int top_k);
    std::vector<int>     find_nearest_points(int centroid_idx, int top_k);
};

// ── MPI — unchanged interface ─────────────────────────────────────────────────
class MPIKMeans : public KNNAlgorithm {
  private:
    int rank_;
    int size_;
    int global_n_;

    std::vector<TVector> local_points_;
    std::vector<int>     local_membership_;
    std::vector<TVector> centroids_;

  public:
    MPIKMeans(Dataset *dataset, int num_clusters, int rank, int size);

    void create_clusters();
    void update_centroids();
    int  find_nearest_centroid(TVector &point);
    std::vector<TVector> query_clusters(TVector &query, int top_k);
    std::vector<int>     find_nearest_points(int centroid_idx, int top_k);
};

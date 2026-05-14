#pragma once

#include "rolodex/dataset.hpp"
#include "rolodex/query_result.hpp"
#include "rolodex/types.hpp"

#include <iosfwd>
#include <mpi.h>
#include <string>
#include <vector>

class KNNAlgorithm {
  protected:
    static const char *cache_root_dir();
    Dataset *dataset_;
    int num_clusters_;
    bool cache_enabled_;
    std::string build_cache_path(const char *algorithm_name) const;
    bool ensure_cache_root_dir() const;

  public:
    KNNAlgorithm(Dataset *dataset, int num_clusters, bool cache_enabled);
    virtual ~KNNAlgorithm() = default;

    /** Centroid updates run when `iter % update_frequency == 0` after membership assignment, unless
     * membership changes are zero (then stop). */
    virtual void create_clusters(int update_frequency = 1) = 0;

    virtual QueryResult query_clusters(const TVector &query, int top_k, int nprobe) const = 0;

    virtual void print_cluster_build_metrics(std::ostream &out) const = 0;
};

// ── Serial — flat memory layout ───────────────────────────────────────────────
class SerialKNNAlgorithm : public KNNAlgorithm {
  public:
    SerialKNNAlgorithm(Dataset *dataset, int num_clusters, bool cache_enabled);

    void create_clusters(int update_frequency = 1) override;

    void print_cluster_build_metrics(std::ostream &out) const override;

    void update_centroids();

    int find_nearest_centroid(const float *query_point) const;

    QueryResult query_clusters(const TVector &query, int top_k, int nprobe) const override;

  private:
    TAlignedVector flat_centroids_;
    std::vector<int> membership_;
    TAlignedVector centroid_sums_;
    TAlignedVector centroid_counts_;
    std::string get_cache_path() const;
    std::size_t dimension_;
    bool load_clusters_from_cache();
    void save_clusters_to_cache() const;
    double cluster_membership_ms_ = 0.0;
    double cluster_centroid_update_ms_ = 0.0;
    std::size_t cluster_membership_iters_ = 0;
};

// ── OpenMP — unchanged interface ──────────────────────────────────────────────
class OpenMPKNNAlgorithm : public KNNAlgorithm {
  public:
    OpenMPKNNAlgorithm(Dataset *dataset, int num_clusters, bool cache_enabled,
                       bool debug_enabled = false);

    void create_clusters(int update_frequency = 1) override;

    void print_cluster_build_metrics(std::ostream &out) const override;

    void update_centroids();

    QueryResult query_clusters(const TVector &query, int top_k, int nprobe) const override;

  private:
    static const char *debug_root_dir();
    std::vector<float> flat_centroids_;
    std::vector<int> membership_;
    bool debug_enabled_;
    std::size_t dimension_;

    int find_nearest_centroid(const float *point) const;
    bool ensure_debug_root_dir() const;
    std::string build_debug_snapshot_path(int iteration, bool is_final) const;
    void save_debug_snapshot(int iteration, bool is_final) const;
    double cluster_membership_ms_ = 0.0;
    double cluster_centroid_update_ms_ = 0.0;
    std::size_t cluster_membership_iters_ = 0;
};

// ── MPI — conforms to KNNAlgorithm virtual interface ──────────────────────────
class MPIKMeans : public KNNAlgorithm {
  private:
    int rank_;
    int size_;
    int global_n_;
    /** Global point index of `local_points_[0]` in the full dataset ordering used at scatter. */
    int global_point_offset_;

    std::vector<float> local_points_flat_;
    std::vector<int> local_membership_;
    std::vector<float> flat_centroids_;
    int dimension_;

    void update_centroids();
    int find_nearest_centroid(const float *point) const;

    double cluster_membership_ms_ = 0.0;
    double cluster_centroid_update_ms_ = 0.0;
    double cluster_mpi_membership_comm_ms_ = 0.0;
    double cluster_mpi_centroid_comm_ms_ = 0.0;
    std::size_t cluster_membership_iters_ = 0;

    bool partition_train_ = false;

  public:
    MPIKMeans(Dataset *dataset, int num_clusters, bool cache_enabled, int rank, int size,
              bool partition_train = false);

    void create_clusters(int update_frequency = 1) override;
    void print_cluster_build_metrics(std::ostream &out) const override;
    QueryResult query_clusters(const TVector &query, int top_k, int nprobe) const override;

    /** Vector dimension (same as centroid width); valid after `create_clusters`. */
    int vector_dim() const;
};

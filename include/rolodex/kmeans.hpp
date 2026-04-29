#pragma once

#include "rolodex/dataset.hpp"
#include "rolodex/query_result.hpp"
#include "rolodex/types.hpp"

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

    /** Serial ignores `update_frequency`; OpenMP uses it for centroid update cadence. */
    virtual void create_clusters(int update_frequency = 1) = 0;

    virtual QueryResult query_clusters(const TVector &query, int top_k, int nprobe) const = 0;
};

// ── Serial — flat memory layout ───────────────────────────────────────────────
class SerialKNNAlgorithm : public KNNAlgorithm {
  public:
    SerialKNNAlgorithm(Dataset *dataset, int num_clusters, bool cache_enabled);

    void create_clusters(int update_frequency = 1) override;

    void update_centroids();

    int find_nearest_centroid(const TVector &point) const;

    QueryResult query_clusters(const TVector &query, int top_k, int nprobe) const override;

  private:
    std::vector<TVector> centroids_;
    std::vector<int> membership_;
    std::string get_cache_path() const;
    bool load_clusters_from_cache();
    void save_clusters_to_cache() const;
    std::vector<int> find_nearest_points(int centroid_idx, int top_k) const;
};

// ── OpenMP — unchanged interface ──────────────────────────────────────────────
class OpenMPKNNAlgorithm : public KNNAlgorithm {
  public:
    OpenMPKNNAlgorithm(Dataset *dataset, int num_clusters, bool cache_enabled,
                       bool debug_enabled = false);

    void create_clusters(int update_frequency = 1) override;

    void update_centroids();

    QueryResult query_clusters(const TVector &query, int top_k, int nprobe) const override;

  private:
    static const char *debug_root_dir();
    std::vector<TVector> centroids_;
    std::vector<int> membership_;
    bool debug_enabled_;

    int find_nearest_centroid(const TVector &point) const;
    std::vector<int> find_nearest_points(int centroid_idx, int top_k) const;
    bool ensure_debug_root_dir() const;
    std::string build_debug_snapshot_path(int iteration, bool is_final) const;
    void save_debug_snapshot(int iteration, bool is_final) const;
};

// ── MPI — conforms to KNNAlgorithm virtual interface ──────────────────────────
class MPIKMeans : public KNNAlgorithm {
  private:
    int rank_;
    int size_;
    int global_n_;

    std::vector<TVector> local_points_;
    std::vector<int> local_membership_;
    std::vector<TVector> centroids_;

    void update_centroids();
    int find_nearest_centroid(const TVector &point) const;
    std::vector<int> find_nearest_points(int centroid_idx, int top_k) const;

  public:
    MPIKMeans(Dataset *dataset, int num_clusters, bool cache_enabled, int rank, int size);

    void create_clusters(int update_frequency = 1) override;
    QueryResult query_clusters(const TVector &query, int top_k, int nprobe) const override;
};

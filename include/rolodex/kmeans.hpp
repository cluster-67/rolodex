#pragma once

#include "rolodex/dataset.hpp"
#include "rolodex/query_result.hpp"
#include "rolodex/types.hpp"

#include <vector>

class KNNAlgorithm {
  protected:
    Dataset *dataset_;
    int num_clusters_;

  public:
    KNNAlgorithm(Dataset *dataset, int num_clusters);

    virtual ~KNNAlgorithm() = default;
};

class SerialKNNAlgorithm : public KNNAlgorithm {
  public:
    SerialKNNAlgorithm(Dataset *dataset, int num_clusters);

    void create_clusters();

    void update_centroids();

    int find_nearest_centroid(const TVector &point) const;

    /** Approximate kNN: search points in the `nprobe` nearest centroids, return `top_k` closest by
     * squared L2. */
    QueryResult query_clusters(const TVector &query, int top_k, int nprobe) const;

  private:
    std::vector<TVector> centroids_;
    std::vector<int> membership_;
    std::vector<int> find_nearest_points(int centroid_idx, int top_k) const;
};

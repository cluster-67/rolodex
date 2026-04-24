#pragma once

#include "rolodex/dataset.hpp"
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
  private:
    std::vector<TVector> centroids_;
    std::vector<int> membership_;

  public:
    SerialKNNAlgorithm(Dataset *dataset, int num_clusters);

    void create_clusters();

    void update_centroids();

    int find_nearest_centroid(TVector &point);

    std::vector<TVector> query_clusters(TVector &query, int top_k);

    std::vector<int> find_nearest_points(int centroid_idx, int top_k);
};

class OpenMPKNNAlgorithm : public KNNAlgorithm {
  private:
    std::vector<TVector> centroids_;
    std::vector<int> membership_;

  public:
    OpenMPKNNAlgorithm(Dataset *dataset, int num_clusters);

    void create_clusters(int update_frequency);

    void update_centroids();

    int find_nearest_centroid(TVector &point);

    std::vector<TVector> query_clusters(TVector &query, int top_k);

    std::vector<int> find_nearest_points(int centroid_idx, int top_k);
};

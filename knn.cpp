#include <H5Cpp.h>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>
using namespace std;
using namespace H5;

// static vector<vector<int>> clusters;
// static vector<int> membership;
// static vector<vector<int>> points;

using TVector = vector<float>;

// Load dataset from hd5 file
class Dataset {
private:
  string filename;
  vector<TVector> points;

public:
  Dataset(string filename) { this->filename = filename; }

  void load_dataset() {
    // Open file
    H5File file(this->filename, H5F_ACC_RDONLY);

    // Open /train dataset and inspect shape.
    DataSet dataset = file.openDataSet("/train");
    DataSpace dataspace = dataset.getSpace();
    int rank = dataspace.getSimpleExtentNdims();
    if (rank != 2) {
      cerr << "Expected /train to be rank-2, got rank " << rank << endl;
      return;
    }

    hsize_t dims[2];
    dataspace.getSimpleExtentDims(dims);
    const size_t nrows = static_cast<size_t>(dims[0]);
    const size_t ncols = static_cast<size_t>(dims[1]);

    // Read contiguous dataset buffer and convert to vector<vector<int>>.
    vector<float> raw(nrows * ncols);
    dataset.read(raw.data(), PredType::NATIVE_FLOAT);

    this->points.assign(nrows, TVector(ncols));
    for (size_t i = 0; i < nrows; ++i) {
      for (size_t j = 0; j < ncols; ++j) {
        this->points[i][j] = raw[i * ncols + j];
      }
    }

    cout << "Loaded /train into points with shape " << nrows << "x" << ncols
         << endl;
  }

  vector<TVector> &get_points() { return this->points; }
};

// KNN Algo impl class which takes in a pointer to a dataset object
class KNNAlgorithm {
protected:
  Dataset *dataset;
  int num_clusters;

public:
  KNNAlgorithm(Dataset *dataset, int num_clusters)
      : dataset(dataset), num_clusters(num_clusters) {}
  void create_clusters();
  void query_clusters();
};

class SerialKNNAlgorithm : public KNNAlgorithm {
private:
  vector<TVector> centroids;
  vector<int> membership;

public:
  SerialKNNAlgorithm(Dataset *dataset, int num_clusters)
      : KNNAlgorithm(dataset, num_clusters) {
    this->centroids.resize(num_clusters);
    // init -1 for each point
    this->membership.resize(dataset->get_points().size(), -1);
  }
  void create_clusters() {
    vector<TVector> &points = this->dataset->get_points();
    // randomly assign points to centroids
    for (int c_idx = 0; c_idx < this->num_clusters; c_idx++) {
      int point_idx = rand() % points.size();
      this->centroids[c_idx] = points[point_idx];
      this->membership[point_idx] = c_idx;
    }

    int iters = 0;
    // iterate until convergence
    while (true) {
      iters++;
      int membership_change_count = 0;
      // assign points to nearest centroid
      for (int point_idx = 0; point_idx < points.size(); point_idx++) {
        int nearest_centroid_idx =
            this->find_nearest_centroid(points[point_idx]);
        int prev_membership = this->membership[point_idx];
        if (prev_membership != nearest_centroid_idx) {
          this->membership[point_idx] = nearest_centroid_idx;
          membership_change_count++;
        }
      }

      cout << "Iteration " << iters << " with " << membership_change_count
           << " membership changes" << endl;
      if (membership_change_count == 0) {
        break;
      }

      // update centroids
      this->update_centroids();
    }
  }

  // iterate over membership vector and maintain running sum of points for each
  // centroid, calc all centroid means after
  void update_centroids() {
    vector<TVector> centroid_sums(
        this->num_clusters, TVector(this->dataset->get_points()[0].size(), 0));
    vector<float> centroid_counts(this->num_clusters, 0);
    vector<TVector> &points = this->dataset->get_points();
    for (int point_idx = 0; point_idx < points.size(); point_idx++) {
      int centroid_idx = this->membership[point_idx];
      for (int i = 0; i < points[point_idx].size(); i++) {
        centroid_sums[centroid_idx][i] += points[point_idx][i];
      }
      centroid_counts[centroid_idx]++;
    }
    for (int c_idx = 0; c_idx < this->num_clusters; c_idx++) {
      for (int i = 0; i < centroid_sums[c_idx].size(); i++) {
        centroid_sums[c_idx][i] /= static_cast<float>(centroid_counts[c_idx]);
      }
      this->centroids[c_idx] = centroid_sums[c_idx];
    }
  }

  int find_nearest_centroid(TVector &point) {
    int nearest_centroid_idx = 0;
    float nearest_distance = distance(point, this->centroids[0]);
    for (int c_idx = 1; c_idx < this->num_clusters; c_idx++) {
      float distance = this->distance(point, this->centroids[c_idx]);
      if (distance < nearest_distance) {
        nearest_distance = distance;
        nearest_centroid_idx = c_idx;
      }
    }
    return nearest_centroid_idx;
  }

  float distance(TVector &a, TVector &b) {
    float distance = 0;
    for (int i = 0; i < a.size(); i++) {
      distance += (a[i] - b[i]) * (a[i] - b[i]);
    }
    return sqrtf(distance);
  }

  vector<TVector> query_clusters(TVector &query, int top_k) {
    int nearest_centroid_idx = this->find_nearest_centroid(query);
    vector<int> nearest_points_indices =
        this->find_nearest_points(nearest_centroid_idx, top_k);
    vector<TVector> nearest_points;
    for (int i = 0; i < nearest_points_indices.size(); i++) {
      nearest_points.push_back(
          this->dataset->get_points()[nearest_points_indices[i]]);
    }
    return nearest_points;
  }

  vector<int> find_nearest_points(int centroid_idx, int top_k) {
    vector<int> nearest_points_indices;
    for (int i = 0; i < this->dataset->get_points().size(); i++) {
      if (this->membership[i] == centroid_idx) {
        nearest_points_indices.push_back(i);
      }
    }
    return nearest_points_indices;
  }
};

class Benchmark {
public:
  Benchmark(Dataset *dataset, KNNAlgorithm *knn_algorithm, int top_k);

  // create and query clusters then verify the results
  void run();
  void print_stats();
};

int main(int argc, char **argv) {
  string DATASET_FILE =
      "/pscratch/sd/a/ac3354/data/fashion-mnist-784-euclidean.hdf5";

  Dataset *dataset = new Dataset(DATASET_FILE);
  dataset->load_dataset();

  SerialKNNAlgorithm *knn_algorithm = new SerialKNNAlgorithm(dataset, 10);
  knn_algorithm->create_clusters();

  return 0;
}

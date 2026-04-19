#include "rolodex/dataset.hpp"
#include "rolodex/kmeans.hpp"

#include <string>

int main(int argc, char **argv) {
    (void)argc;
    (void)argv;

    const std::string dataset_file = "/pscratch/sd/a/ac3354/data/fashion-mnist-784-euclidean.hdf5";

    Dataset dataset(dataset_file);
    dataset.load_dataset();

    SerialKNNAlgorithm knn_algorithm(&dataset, 10);
    knn_algorithm.create_clusters();

    return 0;
}

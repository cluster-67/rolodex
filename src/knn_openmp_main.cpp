#include "rolodex/dataset.hpp"
#include "rolodex/kmeans.hpp"

#include <string>

const std::string usr = "asv48";

int main(int argc, char **argv) {
    (void)argc;
    (void)argv;

    const std::string dataset_file = "/pscratch/sd/a/" + usr + "/data/fashion-mnist-784-euclidean.hdf5";

    Dataset dataset(dataset_file);
    dataset.load_dataset();

    OpenMPKNNAlgorithm knn_algorithm(&dataset, 10);
    knn_algorithm.create_clusters();

    return 0;
}

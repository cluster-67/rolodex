#include "rolodex/dataset.hpp"
#include "rolodex/kmeans.hpp"

#include <string>

const std::string usr = "asv48";

int main(int argc, char **argv) {
    const int update_frequency = (argc > 1) ? std::stoi(argv[1]) : 1;

    const std::string dataset_file = "/pscratch/sd/a/" + usr + "/data/fashion-mnist-784-euclidean.hdf5";

    Dataset dataset(dataset_file);
    dataset.load_dataset();

    OpenMPKNNAlgorithm knn_algorithm(&dataset, 10);
    knn_algorithm.create_clusters(update_frequency);

    return 0;
}

#include "rolodex/dataset.hpp"
#include "rolodex/kmeans.hpp"

#include <iostream>
#include <string>

int main(int argc, char **argv) {
    (void)argc;
    (void)argv;

    const std::string dataset_file = "/pscratch/sd/a/ac3354/data/fashion-mnist-784-euclidean.hdf5";

    Dataset dataset(dataset_file);
    dataset.load_dataset();

    SerialKNNAlgorithm knn_algorithm(&dataset, 10);
    knn_algorithm.create_clusters();

    try {
        dataset.load_validation_dataset(3);
    } catch (const std::exception &e) {
        std::cerr << "Failed to load validation dataset: " << e.what() << '\n';
        return 1;
    }
    auto validation_points = dataset.get_validation_points();
    for (const auto &vp : validation_points) {
        std::cout << vp.to_string(5) << '\n';
    }
    // neighbours = knn_algorithm.find_top_k(query, top_k);
    // validate neighbours - recall?

    return 0;
}

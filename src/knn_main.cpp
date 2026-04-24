#include "rolodex/dataset.hpp"
#include "rolodex/kmeans.hpp"
#include "rolodex/timing.hpp"
#include "rolodex/validator.hpp"

#include <iostream>
#include <memory>

namespace {

enum class RunImplementation {
    Serial,
    OpenMP,
};

struct RunConfig {
    RunImplementation implementation;
    /** OpenMP centroid update cadence; ignored for Serial. */
    int update_frequency;
    const char *dataset_file;
    int num_clusters;
    int top_k;
    int nprobe;
    /** -1 loads all validation queries (clamped in Dataset to file size). */
    int validation_count;
    float vector_match_eps;
};

constexpr RunConfig kRunConfig = {
    RunImplementation::OpenMP,
    1,
    "/pscratch/sd/a/ac3354/data/fashion-mnist-784-euclidean.hdf5",
    10,
    5,
    3,
    10,
    1e-4f,
};

} // namespace

int main() {
    if (kRunConfig.num_clusters <= 0 || kRunConfig.top_k <= 0 || kRunConfig.nprobe <= 0) {
        std::cerr << "RunConfig: num_clusters, top_k, and nprobe must be positive\n";
        return 1;
    }
    if (kRunConfig.update_frequency <= 0) {
        std::cerr << "RunConfig: update_frequency must be positive\n";
        return 1;
    }

    Dataset dataset(kRunConfig.dataset_file);
    dataset.load_dataset();

    std::unique_ptr<KNNAlgorithm> knn_algorithm;
    switch (kRunConfig.implementation) {
    case RunImplementation::Serial:
        knn_algorithm.reset(new SerialKNNAlgorithm(&dataset, kRunConfig.num_clusters));
        break;
    case RunImplementation::OpenMP:
        knn_algorithm.reset(new OpenMPKNNAlgorithm(&dataset, kRunConfig.num_clusters));
        break;
    }

    const auto cluster_build_start = rolodex::timing::SteadyClock::now();
    knn_algorithm->create_clusters(kRunConfig.update_frequency);
    const auto cluster_build_end = rolodex::timing::SteadyClock::now();
    const double cluster_build_ms =
        rolodex::timing::millis_between(cluster_build_start, cluster_build_end);
    std::cout << "cluster_build_time_ms=" << cluster_build_ms << '\n';

    try {
        dataset.load_validation_dataset(kRunConfig.validation_count);
    } catch (const std::exception &e) {
        std::cerr << "Failed to load validation dataset: " << e.what() << '\n';
        return 1;
    }

    const Validator validator(
        dataset, *knn_algorithm,
        ValidatorConfig{kRunConfig.top_k, kRunConfig.nprobe, kRunConfig.vector_match_eps});
    try {
        (void)validator.run(std::cout, std::cerr);
    } catch (const std::exception &e) {
        std::cerr << "Validation failed: " << e.what() << '\n';
        return 1;
    }

    return 0;
}

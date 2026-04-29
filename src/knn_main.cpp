#include "rolodex/cli.hpp"
#include "rolodex/dataset.hpp"
#include "rolodex/kmeans.hpp"
#include "rolodex/timing.hpp"
#include "rolodex/utils.hpp"
#include "rolodex/validator.hpp"

#include <iostream>
#include <memory>
#include <stdexcept>

#include <mpi.h>

namespace {

struct MpiContext {
    bool enabled = false;
    int rank = 0;
    int size = 1;
};

std::string dataset_path_from_name(const std::string &dataset_name) {
    if (dataset_name == "fashion-mnist") {
        return utils::path::dataset_path("fashion-mnist-784-euclidean.hdf5");
    }
    if (dataset_name == "gist") {
        return utils::path::dataset_path("gist-960-euclidean.hdf5");
    }
    if (dataset_name == "mnist") {
        return utils::path::dataset_path("mnist-784-euclidean.hdf5");
    }
    if (dataset_name == "sift") {
        return utils::path::dataset_path("sift-128-euclidean.hdf5");
    }
    throw std::runtime_error("Invalid dataset '" + dataset_name +
                             "'. Expected one of: fashion-mnist, gist, mnist, sift.");
}

} // namespace

int main(int argc, char **argv) {
    const bool want_mpi = rolodex::cli::argv_requests_mpi(argc, argv);

    MpiContext mpi;
    if (want_mpi) {
        mpi.enabled = true;
        MPI_Init(&argc, &argv);
        MPI_Comm_rank(MPI_COMM_WORLD, &mpi.rank);
        MPI_Comm_size(MPI_COMM_WORLD, &mpi.size);
    }

    rolodex::cli::RunConfig cfg;
    try {
        const rolodex::cli::ParseResult parse = rolodex::cli::parse_args(argc, argv, cfg);
        if (parse.should_exit) {
            if (mpi.enabled) {
                MPI_Finalize();
            }
            return parse.exit_code;
        }
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        if (mpi.enabled) {
            MPI_Finalize();
        }
        return 1;
    }

    const std::string dataset_file = dataset_path_from_name(cfg.dataset_file);
    Dataset dataset(dataset_file);
    if (!mpi.enabled || mpi.rank == 0) {
        const auto dataset_load_start = rolodex::timing::SteadyClock::now();

        dataset.load_dataset();

        const double dataset_load_ms = rolodex::timing::millis_between(
            dataset_load_start, rolodex::timing::SteadyClock::now());
        std::cout << "train_dataset_load_time_ms=" << dataset_load_ms << '\n';
    }

    std::unique_ptr<KNNAlgorithm> knn_algorithm;
    switch (cfg.implementation) {
    case rolodex::cli::RunImplementation::Serial:
        knn_algorithm.reset(new SerialKNNAlgorithm(&dataset, cfg.num_clusters, cfg.cache_enabled));
        break;
    case rolodex::cli::RunImplementation::OpenMP:
        knn_algorithm.reset(new OpenMPKNNAlgorithm(&dataset, cfg.num_clusters, cfg.cache_enabled));
        break;
    case rolodex::cli::RunImplementation::MPI:
        knn_algorithm.reset(
            new MPIKMeans(&dataset, cfg.num_clusters, cfg.cache_enabled, mpi.rank, mpi.size));
        break;
    }

    if (mpi.enabled) {
        MPI_Barrier(MPI_COMM_WORLD);
    }
    const auto cluster_build_start = rolodex::timing::SteadyClock::now();
    knn_algorithm->create_clusters(cfg.update_frequency);
    const auto cluster_build_end = rolodex::timing::SteadyClock::now();
    if (mpi.enabled) {
        MPI_Barrier(MPI_COMM_WORLD);
    }

    if (!mpi.enabled || mpi.rank == 0) {
        const double cluster_build_ms =
            rolodex::timing::millis_between(cluster_build_start, cluster_build_end);
        std::cout << "cluster_build_time_ms=" << cluster_build_ms << '\n';
    }

    if (!mpi.enabled || mpi.rank == 0) {
        try {
            const auto validation_load_start = rolodex::timing::SteadyClock::now();

            dataset.load_validation_dataset(cfg.validation_count);

            const double validation_load_ms = rolodex::timing::millis_between(
                validation_load_start, rolodex::timing::SteadyClock::now());
            std::cout << "validation_dataset_load_time_ms=" << validation_load_ms << '\n';
        } catch (const std::exception &e) {
            std::cerr << "Failed to load validation dataset: " << e.what() << '\n';
            if (mpi.enabled) {
                MPI_Finalize();
            }
            return 1;
        }

        const Validator validator(dataset, *knn_algorithm,
                                  ValidatorConfig{cfg.top_k, cfg.nprobe, cfg.vector_match_eps});
        try {
            (void)validator.run(std::cout, std::cerr);
        } catch (const std::exception &e) {
            std::cerr << "Validation failed: " << e.what() << '\n';
            if (mpi.enabled) {
                MPI_Finalize();
            }
            return 1;
        }
    }

    if (mpi.enabled) {
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Finalize();
    }
    return 0;
}

#include "rolodex/cli.hpp"
#include "rolodex/dataset.hpp"
#include "rolodex/kmeans.hpp"
#include "rolodex/timing.hpp"
#include "rolodex/utils.hpp"
#include "rolodex/validator.hpp"

#include <iostream>
#include <memory>

#include <mpi.h>

namespace {

struct MpiContext {
    bool enabled = false;
    int rank = 0;
    int size = 1;
};

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

    Dataset dataset(cfg.dataset_file);
    if (!mpi.enabled || mpi.rank == 0) {
        dataset.load_dataset();
    }

    std::unique_ptr<KNNAlgorithm> knn_algorithm;
    switch (cfg.implementation) {
    case rolodex::cli::RunImplementation::Serial:
        knn_algorithm.reset(new SerialKNNAlgorithm(&dataset, cfg.num_clusters));
        break;
    case rolodex::cli::RunImplementation::OpenMP:
        knn_algorithm.reset(new OpenMPKNNAlgorithm(&dataset, cfg.num_clusters));
        break;
    case rolodex::cli::RunImplementation::MPI:
        knn_algorithm.reset(new MPIKMeans(&dataset, cfg.num_clusters, mpi.rank, mpi.size));
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
            dataset.load_validation_dataset(cfg.validation_count);
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

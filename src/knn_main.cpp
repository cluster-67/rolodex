#include "rolodex/cli.hpp"
#include "rolodex/dataset.hpp"
#include "rolodex/kmeans.hpp"
#include "rolodex/timing.hpp"
#include "rolodex/utils.hpp"
#include "rolodex/validation.hpp"
#include "rolodex/validator.hpp"

#include <cstdlib>
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
    if (dataset_name == "deep1b-100") {
        return utils::path::dataset_path("deep1b/deep1b-n100.hdf5");
    }
    if (dataset_name == "deep1b-1K") {
        return utils::path::dataset_path("deep1b/deep1b-n1000.hdf5");
    }
    if (dataset_name == "deep1b-10K") {
        return utils::path::dataset_path("deep1b/deep1b-n10000.hdf5");
    }
    if (dataset_name == "deep1b-100K") {
        return utils::path::dataset_path("deep1b/deep1b-n100000.hdf5");
    }
    if (dataset_name == "deep1b-1M") {
        return utils::path::dataset_path("deep1b/deep1b-n1000000.hdf5");
    }
    if (dataset_name == "deep1b-10M") {
        return utils::path::dataset_path("deep1b/deep1b-n10000000.hdf5");
    }
    if (dataset_name == "deep1b-100M") {
        return utils::path::dataset_path("deep1b/deep1b-n100000000.hdf5");
    }
    if (dataset_name == "deep1b-1B") {
        return utils::path::dataset_path("deep1b/deep1b-n1000000000.hdf5");
    }
    throw std::runtime_error("Invalid dataset '" + dataset_name +
                             "'. Expected one of: fashion-mnist, gist, mnist, sift.");
}

} // namespace

namespace {

ValidationSummary run_mpi_collective_validation(Dataset &dataset, MPIKMeans &mpi_algo,
                                                const ValidatorConfig &vcfg, int rank,
                                                std::ostream &out, std::ostream &err) {
    const std::vector<ValidationPoint> *vps = nullptr;
    int nq = 0;
    if (rank == 0) {
        vps = &dataset.get_validation_points();
        if (vps->empty()) {
            throw std::runtime_error("No validation points loaded");
        }
        nq = static_cast<int>(vps->size());
    }
    MPI_Bcast(&nq, 1, MPI_INT, 0, MPI_COMM_WORLD);

    rolodex::timing::QueryLatencyAccumulator query_latency;
    rolodex::timing::QueryStageTimings stage_totals;
    float recall_sum = 0.0f;

    const int dim = mpi_algo.vector_dim();
    TVector q_dummy(static_cast<std::size_t>(dim > 0 ? dim : 0), 0.0f);

    rolodex::timing::set_query_stage_sink(&stage_totals);
    for (int qi = 0; qi < nq; ++qi) {
        const auto q_start = rolodex::timing::SteadyClock::now();
        QueryResult predicted;
        if (rank == 0) {
            predicted = mpi_algo.query_clusters((*vps)[static_cast<std::size_t>(qi)].query,
                                                vcfg.top_k, vcfg.nprobe);
        } else {
            predicted = mpi_algo.query_clusters(q_dummy, vcfg.top_k, vcfg.nprobe);
        }
        const auto q_end = rolodex::timing::SteadyClock::now();
        const double q_ms = rolodex::timing::millis_between(q_start, q_end);

        if (rank == 0) {
            query_latency.add_sample(q_ms);
            const float recall = rolodex::validation::recall_at_k(
                (*vps)[static_cast<std::size_t>(qi)], predicted, vcfg.top_k, vcfg.vector_match_eps);
            recall_sum += recall;
            out << "query=" << qi << " top_k=" << vcfg.top_k << " nprobe=" << vcfg.nprobe
                << " recall@" << vcfg.top_k << "=" << recall << " query_time_ms=" << q_ms << '\n';
            if (recall < 1.0f - 1e-6f) {
                err << "query=" << qi << " recall below 1.0; diagnostics:\n";
                rolodex::validation::print_miss_diagnostics((*vps)[static_cast<std::size_t>(qi)],
                                                            predicted, vcfg.top_k,
                                                            vcfg.vector_match_eps, err);
            }
        }
    }
    rolodex::timing::set_query_stage_sink(nullptr);

    auto print_rank_metric = [&](const char *key, double value) {
        out << "rank=" << rank << ' ' << key << '=' << value << '\n';
    };
    print_rank_metric("query_centroid_dist_ms", stage_totals.centroid_dist_ms);
    print_rank_metric("query_scan_ms", stage_totals.scan_ms);
    print_rank_metric("query_mpi_bcast_ms", stage_totals.mpi_bcast_ms);
    print_rank_metric("query_mpi_gather_ms", stage_totals.mpi_gather_ms);
    print_rank_metric("query_mpi_merge_ms", stage_totals.mpi_merge_ms);
    print_rank_metric("query_result_assemble_ms", stage_totals.result_assemble_ms);

    if (rank == 0) {
        const float mean_recall = recall_sum / static_cast<float>(nq);
        out << "aggregate: mean_recall@" << vcfg.top_k << "=" << mean_recall << '\n';
        query_latency.print_aggregate(out);
        out << "validation_query_count=" << nq << '\n';
        return ValidationSummary{static_cast<std::size_t>(nq), mean_recall};
    }
    return ValidationSummary{0, 0.0f};
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

    std::srand(cfg.seed);

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
        knn_algorithm.reset(new OpenMPKNNAlgorithm(&dataset, cfg.num_clusters, cfg.cache_enabled,
                                                   cfg.debug_enabled));
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
    knn_algorithm->print_cluster_build_metrics(std::cout);

    const ValidatorConfig vcfg{cfg.top_k, cfg.nprobe, cfg.vector_match_eps};

    if (mpi.enabled && cfg.implementation == rolodex::cli::RunImplementation::MPI) {
        if (mpi.rank == 0) {
            try {
                const auto validation_load_start = rolodex::timing::SteadyClock::now();

                dataset.load_validation_dataset(cfg.validation_count);

                const double validation_load_ms = rolodex::timing::millis_between(
                    validation_load_start, rolodex::timing::SteadyClock::now());
                std::cout << "validation_dataset_load_time_ms=" << validation_load_ms << '\n';
            } catch (const std::exception &e) {
                std::cerr << "Failed to load validation dataset: " << e.what() << '\n';
                MPI_Finalize();
                return 1;
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);

        auto *mpi_knn = dynamic_cast<MPIKMeans *>(knn_algorithm.get());
        if (mpi_knn == nullptr) {
            std::cerr << "Internal error: MPI implementation without MPIKMeans\n";
            MPI_Finalize();
            return 1;
        }
        try {
            (void)run_mpi_collective_validation(dataset, *mpi_knn, vcfg, mpi.rank, std::cout,
                                                std::cerr);
        } catch (const std::exception &e) {
            std::cerr << "Validation failed: " << e.what() << '\n';
            MPI_Finalize();
            return 1;
        }
    } else if (!mpi.enabled || mpi.rank == 0) {
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

        const Validator validator(dataset, *knn_algorithm, vcfg);
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

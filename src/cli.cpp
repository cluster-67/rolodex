#include "rolodex/cli.hpp"

#include "rolodex/third_party/CLI11.hpp"

#include <stdexcept>
#include <string>

namespace rolodex {
namespace cli {

namespace {

const char *to_string(RunImplementation impl) {
    switch (impl) {
    case RunImplementation::Serial:
        return "serial";
    case RunImplementation::OpenMP:
        return "openmp";
    case RunImplementation::MPI:
        return "mpi";
    }
    return "unknown";
}

RunImplementation parse_impl_string(const std::string &s) {
    if (s == "serial") {
        return RunImplementation::Serial;
    }
    if (s == "openmp") {
        return RunImplementation::OpenMP;
    }
    if (s == "mpi") {
        return RunImplementation::MPI;
    }
    throw std::runtime_error("Invalid impl '" + s + "'. Expected one of: serial, openmp, mpi.");
}

} // namespace

bool argv_requests_mpi(int argc, char **argv) {
    if (argc < 2 || argv == nullptr || argv[1] == nullptr) {
        return false;
    }
    return std::string(argv[1]) == "mpi";
}

ParseResult parse_args(int argc, char **argv, RunConfig &out) {
    RunConfig cfg;

    // Defaults (mirror previous `kRunConfig` in src/knn_main.cpp).
    cfg.implementation = RunImplementation::Serial; // overwritten by required positional
    cfg.update_frequency = 1;
    cfg.cache_enabled = false;
    cfg.dataset_file = "fashion-mnist";
    cfg.num_clusters = 10;
    cfg.top_k = 5;
    cfg.nprobe = 1;
    cfg.validation_count = 10;
    cfg.vector_match_eps = 1e-4f;
    cfg.debug_enabled = false;
    cfg.seed = 42U;

    CLI::App app{"rolodex knn"};

    std::string impl_str;
    auto *impl_opt =
        app.add_option("impl", impl_str, "Implementation backend: serial|openmp|mpi")->required();
    impl_opt->check(CLI::IsMember({"serial", "openmp", "mpi"}));

    app.add_option("--dataset", cfg.dataset_file, "Dataset name: fashion-mnist|gist|mnist|sift")
        ->check(CLI::IsMember({"fashion-mnist", "gist", "mnist", "sift"}))
        ->default_val(cfg.dataset_file);

    app.add_option("-k,--num-clusters", cfg.num_clusters, "Number of k-means clusters")
        ->default_val(cfg.num_clusters);
    app.add_option("--top-k", cfg.top_k, "Top-k neighbors to return/validate")
        ->default_val(cfg.top_k);
    app.add_option("--nprobe", cfg.nprobe, "Number of clusters to probe")->default_val(cfg.nprobe);
    app.add_option("--update-frequency", cfg.update_frequency,
                   "Centroid update cadence: update when iter %% update_frequency == 0 (all impls)")
        ->default_val(cfg.update_frequency);
    app.add_option("--seed", cfg.seed, "RNG seed for deterministic centroid initialization (srand)")
        ->default_val(cfg.seed);
    app.add_flag("--cache", cfg.cache_enabled, "Enable cluster cache (disabled by default)");
    app.add_option("--validation-count", cfg.validation_count,
                   "Number of validation queries to load (-1 = all)")
        ->default_val(cfg.validation_count);
    app.add_option("--vector-match-eps", cfg.vector_match_eps,
                   "Epsilon for vector equality in recall computation")
        ->default_val(cfg.vector_match_eps);
    app.add_flag("--debug", cfg.debug_enabled,
                 "Enable OpenMP debug snapshots (HDF5) under data/debug");

    try {
        app.parse(argc, argv);
    } catch (const CLI::ParseError &e) {
        ParseResult r;
        r.exit_code = app.exit(e);
        r.should_exit = true;
        return r;
    }

    cfg.implementation = parse_impl_string(impl_str);

    if (cfg.num_clusters <= 0 || cfg.top_k <= 0 || cfg.nprobe <= 0) {
        throw std::runtime_error("RunConfig: num_clusters, top_k, and nprobe must be positive");
    }
    if (cfg.update_frequency <= 0) {
        throw std::runtime_error("RunConfig: update_frequency must be positive");
    }

    out = cfg;
    (void)to_string; // keep helper around for future logging/debug
    ParseResult r;
    r.exit_code = 0;
    r.should_exit = false;
    return r;
}

} // namespace cli
} // namespace rolodex

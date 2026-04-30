#pragma once

#include <string>

namespace rolodex {
namespace cli {

enum class RunImplementation {
    Serial,
    OpenMP,
    MPI,
};

struct RunConfig {
    RunImplementation implementation;
    /** Centroid update cadence: update when `iter % update_frequency == 0` after a non-zero-change
     * iteration (all implementations). */
    int update_frequency;
    /** Enables cluster cache load/save when supported by the implementation. */
    bool cache_enabled;
    std::string dataset_file;
    int num_clusters;
    int top_k;
    int nprobe;
    /** -1 loads all validation queries (clamped in Dataset to file size). */
    int validation_count;
    float vector_match_eps;
    /** Enables OpenMP cluster debug snapshots under data/debug. */
    bool debug_enabled;
    /** RNG seed for deterministic centroid init (`srand`); used by serial, OpenMP, and MPI. */
    unsigned int seed;
};

/** Bootstrap: determine whether argv requests MPI, without full parsing. */
bool argv_requests_mpi(int argc, char **argv);

struct ParseResult {
    /** Process exit code for CLI outcomes. Meaningful when `should_exit==true`. */
    int exit_code = 0;
    /** True if the caller should exit immediately (help/usage/parse error). */
    bool should_exit = false;
};

/** Parse full CLI (positional impl + named flags). */
ParseResult parse_args(int argc, char **argv, RunConfig &out);

} // namespace cli
} // namespace rolodex

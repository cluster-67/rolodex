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
    /** OpenMP centroid update cadence; ignored by Serial/MPI (current behavior). */
    int update_frequency;
    std::string dataset_file;
    int num_clusters;
    int top_k;
    int nprobe;
    /** -1 loads all validation queries (clamped in Dataset to file size). */
    int validation_count;
    float vector_match_eps;
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

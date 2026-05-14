#pragma once

#include <chrono>
#include <cstddef>
#include <iosfwd>
#include <limits>

namespace rolodex {
namespace timing {

using SteadyClock = std::chrono::steady_clock;
using SteadyTimePoint = SteadyClock::time_point;

inline double millis_between(SteadyTimePoint start, SteadyTimePoint end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

/** Accumulates per-query latencies (milliseconds) and can print aggregate stats. */
class QueryLatencyAccumulator {
  public:
    void add_sample(double ms);

    double total_ms() const {
        return total_ms_;
    }
    double min_ms() const {
        return min_ms_;
    }
    double max_ms() const {
        return max_ms_;
    }
    std::size_t count() const {
        return count_;
    }
    double mean_ms() const;

    void print_aggregate(std::ostream &out) const;

  private:
    double total_ms_ = 0.0;
    double min_ms_ = std::numeric_limits<double>::infinity();
    double max_ms_ = 0.0;
    std::size_t count_ = 0;
};

struct QueryStageTimings {
    double centroid_dist_ms = 0.0;
    double scan_ms = 0.0;
    double openmp_merge_ms = 0.0;
    double mpi_bcast_ms = 0.0;
    double mpi_gather_ms = 0.0;
    double mpi_merge_ms = 0.0;
    double result_assemble_ms = 0.0;

    void add(const QueryStageTimings &other) {
        centroid_dist_ms += other.centroid_dist_ms;
        scan_ms += other.scan_ms;
        openmp_merge_ms += other.openmp_merge_ms;
        mpi_bcast_ms += other.mpi_bcast_ms;
        mpi_gather_ms += other.mpi_gather_ms;
        mpi_merge_ms += other.mpi_merge_ms;
        result_assemble_ms += other.result_assemble_ms;
    }
};

void set_query_stage_sink(QueryStageTimings *sink);
QueryStageTimings *query_stage_sink();

class QueryStageGuard {
  public:
    QueryStageGuard() : sink_(query_stage_sink()) {}
    ~QueryStageGuard() {
        flush();
    }
    void flush() {
        if (sink_ != nullptr) {
            sink_->add(stage_);
            sink_ = nullptr;
        }
    }
    QueryStageTimings stage_;

  private:
    QueryStageTimings *sink_;
};

} // namespace timing
} // namespace rolodex

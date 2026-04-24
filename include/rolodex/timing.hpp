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

} // namespace timing
} // namespace rolodex

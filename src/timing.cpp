#include "rolodex/timing.hpp"

#include <ostream>

namespace rolodex {
namespace timing {

namespace {
thread_local QueryStageTimings *g_query_stage_sink = nullptr;
}

void QueryLatencyAccumulator::add_sample(double ms) {
    total_ms_ += ms;
    if (ms < min_ms_) {
        min_ms_ = ms;
    }
    if (ms > max_ms_) {
        max_ms_ = ms;
    }
    count_++;
}

double QueryLatencyAccumulator::mean_ms() const {
    if (count_ == 0) {
        return 0.0;
    }
    return total_ms_ / static_cast<double>(count_);
}

void QueryLatencyAccumulator::print_aggregate(std::ostream &out) const {
    out << "aggregate: query_time_total_ms=" << total_ms_ << " mean_ms=" << mean_ms()
        << " min_ms=" << min_ms_ << " max_ms=" << max_ms_ << '\n';
}

void set_query_stage_sink(QueryStageTimings *sink) {
    g_query_stage_sink = sink;
}

QueryStageTimings *query_stage_sink() {
    return g_query_stage_sink;
}

} // namespace timing
} // namespace rolodex

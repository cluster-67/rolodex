#include "rolodex/validator.hpp"

#include "rolodex/timing.hpp"
#include "rolodex/validation.hpp"

#include <stdexcept>

Validator::Validator(const Dataset &dataset, const KNNAlgorithm &algorithm, ValidatorConfig config)
    : dataset_(dataset), algorithm_(algorithm), config_(config) {}

ValidationSummary Validator::run(std::ostream &out, std::ostream &err) const {
    const std::vector<ValidationPoint> &validation_points = dataset_.get_validation_points();
    if (validation_points.empty()) {
        throw std::runtime_error("No validation points loaded");
    }

    rolodex::timing::QueryLatencyAccumulator query_latency;
    float recall_sum = 0.0f;

    for (std::size_t qi = 0; qi < validation_points.size(); ++qi) {
        const ValidationPoint &vp = validation_points[qi];

        const auto q_start = rolodex::timing::SteadyClock::now();
        const QueryResult predicted =
            algorithm_.query_clusters(vp.query, config_.top_k, config_.nprobe);
        const auto q_end = rolodex::timing::SteadyClock::now();
        const double q_ms = rolodex::timing::millis_between(q_start, q_end);

        query_latency.add_sample(q_ms);

        const float recall = rolodex::validation::recall_at_k(vp, predicted, config_.top_k,
                                                              config_.vector_match_eps);
        recall_sum += recall;

        out << "query=" << qi << " top_k=" << config_.top_k << " nprobe=" << config_.nprobe
            << " recall@" << config_.top_k << "=" << recall << " query_time_ms=" << q_ms << '\n';

        if (recall < 1.0f - 1e-6f) {
            err << "query=" << qi << " recall below 1.0; diagnostics:\n";
            rolodex::validation::print_miss_diagnostics(vp, predicted, config_.top_k,
                                                        config_.vector_match_eps, err);
        }
    }

    const float mean_recall = recall_sum / static_cast<float>(validation_points.size());
    out << "aggregate: mean_recall@" << config_.top_k << "=" << mean_recall << '\n';
    query_latency.print_aggregate(out);

    return ValidationSummary{validation_points.size(), mean_recall};
}

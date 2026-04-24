#include "rolodex/dataset.hpp"
#include "rolodex/kmeans.hpp"
#include "rolodex/timing.hpp"
#include "rolodex/validation.hpp"

#include <iostream>

namespace {

struct RunConfig {
    const char *dataset_file;
    int num_clusters;
    int top_k;
    int nprobe;
    /** -1 loads all validation queries (clamped in Dataset to file size). */
    int validation_count;
    float vector_match_eps;
};

constexpr RunConfig kRunConfig = {
    "/pscratch/sd/a/ac3354/data/fashion-mnist-784-euclidean.hdf5", 10, 5, 1, 10, 1e-4f,
};

} // namespace

int main() {
    if (kRunConfig.num_clusters <= 0 || kRunConfig.top_k <= 0 || kRunConfig.nprobe <= 0) {
        std::cerr << "RunConfig: num_clusters, top_k, and nprobe must be positive\n";
        return 1;
    }

    Dataset dataset(kRunConfig.dataset_file);
    dataset.load_dataset();

    SerialKNNAlgorithm knn_algorithm(&dataset, kRunConfig.num_clusters);

    const auto cluster_build_start = rolodex::timing::SteadyClock::now();
    knn_algorithm.create_clusters();
    const auto cluster_build_end = rolodex::timing::SteadyClock::now();
    const double cluster_build_ms =
        rolodex::timing::millis_between(cluster_build_start, cluster_build_end);
    std::cout << "cluster_build_time_ms=" << cluster_build_ms << '\n';

    try {
        dataset.load_validation_dataset(kRunConfig.validation_count);
    } catch (const std::exception &e) {
        std::cerr << "Failed to load validation dataset: " << e.what() << '\n';
        return 1;
    }

    const std::vector<ValidationPoint> &validation_points = dataset.get_validation_points();
    if (validation_points.empty()) {
        std::cerr << "No validation points loaded\n";
        return 1;
    }

    rolodex::timing::QueryLatencyAccumulator query_latency;
    float recall_sum = 0.0f;

    for (std::size_t qi = 0; qi < validation_points.size(); ++qi) {
        const ValidationPoint &vp = validation_points[qi];

        const auto q_start = rolodex::timing::SteadyClock::now();
        const QueryResult predicted =
            knn_algorithm.query_clusters(vp.query, kRunConfig.top_k, kRunConfig.nprobe);
        const auto q_end = rolodex::timing::SteadyClock::now();
        const double q_ms = rolodex::timing::millis_between(q_start, q_end);

        query_latency.add_sample(q_ms);

        const float recall = rolodex::validation::recall_at_k(vp, predicted, kRunConfig.top_k,
                                                              kRunConfig.vector_match_eps);
        recall_sum += recall;

        std::cout << "query=" << qi << " recall@" << kRunConfig.top_k << "=" << recall
                  << " query_time_ms=" << q_ms << '\n';

        if (recall < 1.0f - 1e-6f) {
            std::cerr << "query=" << qi << " recall below 1.0; diagnostics:\n";
            rolodex::validation::print_miss_diagnostics(vp, predicted, kRunConfig.top_k,
                                                        kRunConfig.vector_match_eps, std::cerr);
        }
    }

    const double n = static_cast<double>(validation_points.size());
    std::cout << "aggregate: mean_recall@" << kRunConfig.top_k << "="
              << (recall_sum / static_cast<float>(n)) << '\n';
    query_latency.print_aggregate(std::cout);

    return 0;
}

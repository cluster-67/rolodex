#include "rolodex/kmeans.hpp"

#include "rolodex/cache_utils.hpp"

#include <sstream>

KNNAlgorithm::KNNAlgorithm(Dataset *dataset, int num_clusters)
    : dataset_(dataset), num_clusters_(num_clusters) {}

const char *KNNAlgorithm::cache_root_dir() {
    return "data/cluster_cache";
}

uint64_t KNNAlgorithm::dataset_signature() const {
    return cache_utils::compute_dataset_signature(dataset_);
}

std::string KNNAlgorithm::build_cache_path(const char *algorithm_name) const {
    const std::vector<TVector> &points = dataset_->get_points();
    const std::size_t num_points = points.size();
    const std::size_t dim = points.empty() ? 0 : points[0].size();

    std::ostringstream out;
    out << cache_root_dir() << "/cache_" << algorithm_name << "_k" << num_clusters_ << "_n"
        << num_points << "_d" << dim << "_ds" << cache_utils::to_hex(dataset_signature()) << ".bin";
    return out.str();
}

bool KNNAlgorithm::ensure_cache_root_dir() const {
    return cache_utils::ensure_dir_recursive(cache_root_dir(), 0775);
}

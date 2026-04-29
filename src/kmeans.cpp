#include "rolodex/kmeans.hpp"

#include "rolodex/utils.hpp"

#include <sstream>

KNNAlgorithm::KNNAlgorithm(Dataset *dataset, int num_clusters, bool cache_enabled)
    : dataset_(dataset), num_clusters_(num_clusters), cache_enabled_(cache_enabled) {}

const char *KNNAlgorithm::cache_root_dir() {
    return "data/cluster_cache";
}

std::string KNNAlgorithm::build_cache_path(const char *algorithm_name) const {
    const std::vector<TVector> &points = dataset_->get_points();
    const std::size_t num_points = points.size();
    const std::size_t dim = points.empty() ? 0 : points[0].size();
    const std::string basename = utils::cache::dataset_basename(dataset_);

    std::ostringstream out;
    out << cache_root_dir() << "/cache_" << algorithm_name << "_k" << num_clusters_ << "_n"
        << num_points << "_d" << dim << "_dataset_" << basename << ".h5";
    return out.str();
}

bool KNNAlgorithm::ensure_cache_root_dir() const {
    return utils::cache::ensure_dir_recursive(cache_root_dir(), 0775);
}

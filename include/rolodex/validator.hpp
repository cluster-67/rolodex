#pragma once

#include "rolodex/dataset.hpp"
#include "rolodex/kmeans.hpp"

#include <cstddef>
#include <iosfwd>

struct ValidatorConfig {
    int top_k;
    int nprobe;
    float vector_match_eps;
};

struct ValidationSummary {
    std::size_t query_count;
    float mean_recall;
};

class Validator {
  public:
    Validator(const Dataset &dataset, const KNNAlgorithm &algorithm, ValidatorConfig config);

    ValidationSummary run(std::ostream &out, std::ostream &err) const;

  private:
    const Dataset &dataset_;
    const KNNAlgorithm &algorithm_;
    ValidatorConfig config_;
};

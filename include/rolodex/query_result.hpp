#pragma once

#include "rolodex/types.hpp"

#include <vector>

/** Result of an approximate kNN query: neighbor vectors and their distances to the query (same
 * metric as search, e.g. squared L2). */
struct QueryResult {
    std::vector<TVector> neighbors;
    std::vector<float> distances;
};

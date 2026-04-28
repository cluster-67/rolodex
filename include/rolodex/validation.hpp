#pragma once

#include "rolodex/dataset.hpp"
#include "rolodex/query_result.hpp"

#include <iosfwd>

namespace rolodex {
namespace validation {

/** Recall@K: first `k` ground-truth neighbors vs predicted neighbors (vector match with epsilon).
 */
float recall_at_k(const ValidationPoint &vp, const QueryResult &predicted, int k, float eps);

/** When recall < 1, print missed GT (expected distance from file) and extra predictions (received
 * distance from query). */
void print_miss_diagnostics(const ValidationPoint &vp, const QueryResult &predicted, int k,
                            float eps, std::ostream &err);

} // namespace validation
} // namespace rolodex

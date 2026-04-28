#pragma once

#include <cstdint>
#include <string>

class Dataset;

namespace cache_utils {

uint64_t compute_dataset_signature(const Dataset *dataset);
bool ensure_dir_recursive(const std::string &path, int mode);
std::string to_hex(uint64_t value);

} // namespace cache_utils

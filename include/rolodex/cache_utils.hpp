#pragma once

#include <string>

class Dataset;

namespace cache_utils {

bool ensure_dir_recursive(const std::string &path, int mode);
std::string dataset_basename(const Dataset *dataset);

} // namespace cache_utils

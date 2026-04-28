#pragma once

#include <string>

class Dataset;

namespace utils {
namespace cache {

bool ensure_dir_recursive(const std::string &path, int mode);
std::string dataset_basename(const Dataset *dataset);

} // namespace cache

namespace path {

std::string scratch_dir();
std::string dataset_path(const std::string &dataset_filename);

} // namespace path
} // namespace utils
